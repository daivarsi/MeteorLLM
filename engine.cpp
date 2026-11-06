// Qwen3.5 hybrid graph follows llama.cpp 19e8a86fe (MIT, licenses/).
// Reuses MeteorLLM quantized matrix kernels; separate model and device files.
#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <stdexcept>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __vita__
#include <psp2/kernel/sysmem.h>
#endif
namespace vitaqwen{
static uint64_t now(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void req(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
static uint32_t u32(const uint8_t*p){uint32_t v;memcpy(&v,p,4);return v;}
static uint64_t u64(const uint8_t*p){uint64_t v;memcpy(&v,p,8);return v;}
static float dot(const float*a,const float*b,int n){
#ifdef __ARM_NEON
 float32x4_t s=vdupq_n_f32(0);for(int i=0;i<n;i+=4)s=vmlaq_f32(s,vld1q_f32(a+i),vld1q_f32(b+i));float32x2_t t=vadd_f32(vget_low_f32(s),vget_high_f32(s));t=vpadd_f32(t,t);return vget_lane_f32(t,0);
#else
 float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;
#endif
}
static float sigmoid(float x){return 1/(1+std::exp(-x));}
Engine::State::State():x(1024),xb(1024),tmp(1024),q(6144),k(512),v(512),z(2048),alpha(16),beta(16),attn(2048),gate(3584),up(3584){}
Engine::~Engine(){if(!directory.empty())std::remove((directory+"/shared-prefix-app.tmp").c_str());if(emb_file)fclose(emb_file);if(head_file)fclose(head_file);
#ifdef __vita__
 if(mem_uid>=0)sceKernelFreeMemBlock(mem_uid);
#else
 free(model);
#endif
}
const Tensor&Engine::tensor(const std::string&n)const{for(auto&t:tensors)if(t.name==n)return t;throw std::runtime_error("missing "+n);}
void Engine::load(const std::string&dir){directory=dir;FILE*f=fopen((dir+"/weights.bin").c_str(),"rb");req(f,"0.8B model files missing. See release installation notes.");uint8_t h[128];req(fread(h,1,128,f)==128&&!memcmp(h,"VQ350001",8),"header");uint32_t nt=u32(h+48),hs=u32(h+52);model_bytes=u64(h+60);req(nt<400&&hs==((128+nt*96+255)/256)*256&&model_bytes<340ull*1024*1024,"table bounds");
 const uint32_t fields[]={1024,3584,24,8,2,256,248320,512};for(int i=0;i<8;i++)req(u32(h+8+4*i)==fields[i],"unexpected config");
#ifdef __vita__
 mem_uid=sceKernelAllocMemBlock("Overdrive08Weights",SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,(model_bytes+uint64_t(cached_rows)*576+4095)&~4095ull,nullptr);req(mem_uid>=0,"resident model allocation failed");void*p=nullptr;req(sceKernelGetMemBlockBase(mem_uid,&p)>=0,"model mapping");model=(uint8_t*)p;
#else
 model=(uint8_t*)malloc(model_bytes+uint64_t(cached_rows)*576);req(model,"model alloc");
#endif
 rewind(f);req(fread(model,1,model_bytes,f)==model_bytes,"model short read");req(fgetc(f)==EOF,"model extra data");fclose(f);uint64_t cursor=hs;int external=0;
 for(unsigned i=0;i<nt;i++){auto*p=model+128+i*96;req(memchr(p,0,64),"tensor name");Tensor t{(const char*)p,u32(p+64),u32(p+68),u32(p+72),u64(p+76),u64(p+84),nullptr};req(t.cols&&t.cols<=8192&&t.rows&&t.rows<=248320&&(t.type==0||t.type==12||t.type==14),"tensor type/shape");uint64_t expected=uint64_t(t.cols)*t.rows*(t.type==0?4:1);if(t.type)expected=uint64_t(t.cols)*t.rows/256*(t.type==12?144:210);req(t.bytes==expected,"tensor size");
  if(t.offset==UINT64_MAX){req(t.name=="token_embd.weight","external tensor");emb=t;external++;}else{req(t.offset==cursor&&t.bytes<=model_bytes-cursor,"payload bounds");t.data=model+t.offset;cursor+=t.bytes;}tensors.push_back(t);
 }req(cursor==model_bytes&&external==1,"model layout");
 emb_file=fopen((dir+"/embedding.bin").c_str(),"rb");head_file=fopen((dir+"/output-q4-test.bin").c_str(),"rb");req(emb_file&&head_file,"embedding/head files");
 req(!fseek(emb_file,0,SEEK_END)&&ftell(emb_file)==208588800,"embedding size");req(!fseek(head_file,0,SEEK_END)&&ftell(head_file)==143032320,"head size");
 head_cache=model+model_bytes;req(!fseek(head_file,0,SEEK_SET)&&fread(head_cache,576,cached_rows,head_file)==size_t(cached_rows),"output cache read");
 state.resize(18*16*128*128);conv.resize(18*3*6144);keys.resize(6*512*512);values.resize(keys.size());logits.resize(cfg.vocab);io.resize(256*840);for(auto&s:scratch)s.resize(8192);reset();
}
void Engine::reset(){std::fill(state.begin(),state.end(),0);std::fill(conv.begin(),conv.end(),0);next_position=0;reset_metrics();}
void Engine::save_prefix(){req(next_position>0&&next_position<=512,"prefix position");prefix_length=next_position;FILE*f=fopen((directory+"/shared-prefix-app.tmp").c_str(),"wb");req(f,"prefix open");bool ok=fwrite(state.data(),4,state.size(),f)==state.size()&&fwrite(conv.data(),4,conv.size(),f)==conv.size();size_t n=prefix_length*512;for(int l=0;l<6;l++)ok=ok&&fwrite(keys.data()+l*512*512,2,n,f)==n&&fwrite(values.data()+l*512*512,2,n,f)==n;ok=fclose(f)==0&&ok;req(ok,"prefix write");}
void Engine::restore_prefix(){req(prefix_length>0,"prefix missing");FILE*f=fopen((directory+"/shared-prefix-app.tmp").c_str(),"rb");req(f,"prefix read open");bool ok=fread(state.data(),4,state.size(),f)==state.size()&&fread(conv.data(),4,conv.size(),f)==conv.size();size_t n=prefix_length*512;for(int l=0;l<6;l++)ok=ok&&fread(keys.data()+l*512*512,2,n,f)==n&&fread(values.data()+l*512*512,2,n,f)==n;fclose(f);req(ok,"prefix read");next_position=prefix_length;}
void Engine::norm(float*out,const float*in,const Tensor&w,int n){float scale=1/std::sqrt(dot(in,in,n)/n+cfg.eps);auto*weights=(const float*)w.data;for(int i=0;i<n;i++)out[i]=in[i]*scale*weights[i];}
void Engine::embedding(int token,float*out){size_t stride=840;req(!fseek(emb_file,long(token*stride),SEEK_SET)&&fread(io.data(),1,stride,emb_file)==stride,"embedding read");embedding_read_bytes+=stride;dequantize(io.data(),14,out,1024);}
void Engine::matrix(const Tensor&w,State*s,int count,std::vector<float>State::*in,std::vector<float>State::*out){if(cancelled&&cancelled())throw std::runtime_error("Stopped");auto stamp=now();size_t stride=w.bytes/w.rows;std::vector<ActivationBlock>packed;if(int8&&w.type){packed.resize(w.cols/32*count);for(int t=0;t<count;t++)quantize_activation((s[t].*in).data(),packed.data()+t,w.cols,count);}
#ifdef __ARM_NEON
 std::vector<Q4InputTile>tiles;if(int8&&w.type&&count==8){tiles.resize(w.cols/32);for(size_t g=0;g<tiles.size();g++)for(int t=0;t<8;t++){auto&a=packed[g*8+t];tiles[g].scale[t]=a.scale;tiles[g].sum[t]=a.sum;for(int k=0;k<32;k++)tiles[g].lanes[k][t]=a.values[k];}}
#endif
 pool.run(worker_count,w.rows,[&](int first,int last,int worker){for(int r=first;r<last;r++){auto*p=w.data+r*stride;float result[8];
#ifdef __ARM_NEON
 if(!tiles.empty()){(w.type==12?q4_dot_lane8:q6_dot_lane8)(p,tiles.data(),result,w.cols);for(int t=0;t<count;t++)(s[t].*out)[r]=result[t];continue;}
#endif
 if(!packed.empty()){(w.type==12?q4_activation_dot:q6_activation_dot)(p,packed.data(),result,w.cols,count);for(int t=0;t<count;t++)(s[t].*out)[r]=result[t];}
 else{const float*decoded;if(w.type){dequantize(p,w.type,scratch[worker].data(),w.cols);decoded=scratch[worker].data();}else decoded=(const float*)p;for(int t=0;t<count;t++)(s[t].*out)[r]=dot(decoded,(s[t].*in).data(),w.cols);}
 }});matrix_us+=now()-stamp;}
void Engine::linear(State&s,int layer){auto stamp=now();int li=layer-layer/4;std::string b="blk."+std::to_string(layer)+".";auto*kernel=(const float*)tensor(b+"ssm_conv1d.weight").data;float*history=conv.data()+size_t(li)*3*6144;
 for(int c=0;c<6144;c++){float input=s.q[c],v=history[c*3]*kernel[c*4]+history[c*3+1]*kernel[c*4+1]+history[c*3+2]*kernel[c*4+2]+input*kernel[c*4+3];history[c*3]=history[c*3+1];history[c*3+1]=history[c*3+2];history[c*3+2]=input;s.q[c]=v*sigmoid(v);}
 auto*a=(const float*)tensor(b+"ssm_a").data;auto*dt=(const float*)tensor(b+"ssm_dt.bias").data;
 pool.run(worker_count,16,[&](int first,int last,int){for(int h=first;h<last;h++){float*q=s.q.data()+h*128,*k=s.q.data()+2048+h*128,*v=s.q.data()+4096+h*128;float qs=1/std::sqrt(dot(q,q,128)+cfg.eps),ks=1/std::sqrt(dot(k,k,128)+cfg.eps);for(int i=0;i<128;i++){q[i]*=qs;k[i]*=ks;}float alpha=s.alpha[h]+dt[h];float softplus=alpha>20?alpha:std::log1p(std::exp(alpha));float decay=std::exp(a[h]*softplus),beta=sigmoid(s.beta[h]);float*st=state.data()+(size_t(li)*16+h)*128*128;
  for(int j=0;j<128;j++){float*row=st+j*128;
#ifdef __ARM_NEON
   float32x4_t scale=vdupq_n_f32(decay);for(int i=0;i<128;i+=4)vst1q_f32(row+i,vmulq_f32(vld1q_f32(row+i),scale));
#else
   for(int i=0;i<128;i++)row[i]*=decay;
#endif
   float delta=(v[j]-dot(row,k,128))*beta;
#ifdef __ARM_NEON
   float32x4_t d=vdupq_n_f32(delta);for(int i=0;i<128;i+=4)vst1q_f32(row+i,vmlaq_f32(vld1q_f32(row+i),vld1q_f32(k+i),d));
#else
   for(int i=0;i<128;i++)row[i]+=delta*k[i];
#endif
   s.attn[h*128+j]=dot(row,q,128)/std::sqrt(128.f);
  }
  norm(s.attn.data()+h*128,s.attn.data()+h*128,tensor(b+"ssm_norm.weight"),128);for(int i=0;i<128;i++)s.attn[h*128+i]*=s.z[h*128+i]*sigmoid(s.z[h*128+i]);
 }});attention_us+=now()-stamp;}
void Engine::attend(State&s,int layer,int position){auto stamp=now();std::string b="blk."+std::to_string(layer)+".";float query[2048];for(int h=0;h<8;h++)norm(query+h*256,s.q.data()+h*512,tensor(b+"attn_q_norm.weight"),256);for(int h=0;h<2;h++)norm(s.k.data()+h*256,s.k.data()+h*256,tensor(b+"attn_k_norm.weight"),256);
 auto rope=[&](float*p,int heads){for(int h=0;h<heads;h++)for(int i=0;i<32;i++){float angle=position/std::pow(1e7f,2.f*i/64),c=std::cos(angle),sn=std::sin(angle);float x=p[h*256+i],y=p[h*256+i+32];p[h*256+i]=x*c-y*sn;p[h*256+i+32]=x*sn+y*c;}};rope(query,8);rope(s.k.data(),2);
 size_t base=size_t(layer/4)*512*512,slot=base+size_t(position%512)*512;for(int i=0;i<512;i++){keys[slot+i]=float_to_half(s.k[i]);values[slot+i]=float_to_half(s.v[i]);}
 pool.run(worker_count,8,[&](int first,int last,int){for(int h=first;h<last;h++){float scores[512];int n=std::min(position+1,512),oldest=position<512?0:(position+1)%512;float top=-1e30f;for(int p=0;p<n;p++){int slot=(oldest+p)%512;float sum=0;for(int i=0;i<256;i++)sum+=query[h*256+i]*half_to_float(keys[base+slot*512+(h/4)*256+i]);scores[p]=sum/16;top=std::max(top,scores[p]);}float denom=0;for(int p=0;p<n;p++){scores[p]=std::exp(scores[p]-top);denom+=scores[p];}for(int i=0;i<256;i++){float result=0;for(int p=0;p<n;p++)result+=scores[p]/denom*half_to_float(values[base+((oldest+p)%512)*512+(h/4)*256+i]);s.attn[h*256+i]=result*sigmoid(s.q[h*512+256+i]);}}});attention_us+=now()-stamp;}
void Engine::output_head(const float*in){FILE*f=q4_head?head_file:emb_file;size_t stride=q4_head?576:840;std::vector<ActivationBlock>packed;if(q4_head&&int8){packed.resize(32);quantize_activation(in,packed.data(),1024,1);}auto project=[&](int first,int end){for(int base=first;base<end;base+=output_rows){int nr=std::min(output_rows,end-base);if(q4_head&&cache_output&&base<cached_rows)nr=std::min(nr,cached_rows-base);const uint8_t*rows=(q4_head&&cache_output&&base<cached_rows)?head_cache+size_t(base)*stride:io.data();auto stamp=now();if(cancelled&&cancelled())throw std::runtime_error("Stopped");if(rows==io.data()){req(!fseek(f,long(size_t(base)*stride),SEEK_SET)&&fread(io.data(),stride,nr,f)==size_t(nr),"head read");embedding_read_bytes+=stride*nr;}output_read_us+=now()-stamp;stamp=now();pool.run(worker_count,nr,[&](int a,int b,int worker){for(int r=a;r<b;r++){if(!packed.empty())q4_activation_dot(rows+r*stride,packed.data(),logits.data()+base+r,1024,1);else{dequantize(rows+r*stride,q4_head?12:14,scratch[worker].data(),1024);logits[base+r]=dot(scratch[worker].data(),in,1024);}}});output_compute_us+=now()-stamp;}};
 if(!fast_output){project(0,cfg.vocab);output_full_scans++;return;}std::fill(logits.begin()+shortlist,logits.begin()+248044,-1e30f);project(0,shortlist);project(248044,cfg.vocab);float top=-1e30f,second=-1e30f;for(float v:logits){if(v>top){second=top;top=v;}else second=std::max(second,v);}if(top-second<1){project(shortlist,248044);output_full_scans++;}}
const std::vector<float>&Engine::forward_batch(const std::vector<int>&tokens,int position,bool output){req(!tokens.empty()&&tokens.size()<=128&&position==next_position&&position>=0,"batch position/size");
 for(size_t base=0;base<tokens.size();base+=8){int count=std::min(size_t(8),tokens.size()-base);State s[8];for(int t=0;t<count;t++){req(tokens[base+t]>=0&&tokens[base+t]<cfg.vocab,"token bounds");embedding(tokens[base+t],s[t].x.data());}
 for(int l=0;l<24;l++){std::string b="blk."+std::to_string(l)+".";for(int t=0;t<count;t++)norm(s[t].xb.data(),s[t].x.data(),tensor(b+"attn_norm.weight"),1024);
 if((l+1)%4){matrix(tensor(b+"attn_qkv.weight"),s,count,&State::xb,&State::q);matrix(tensor(b+"attn_gate.weight"),s,count,&State::xb,&State::z);matrix(tensor(b+"ssm_alpha.weight"),s,count,&State::xb,&State::alpha);matrix(tensor(b+"ssm_beta.weight"),s,count,&State::xb,&State::beta);for(int t=0;t<count;t++)linear(s[t],l);matrix(tensor(b+"ssm_out.weight"),s,count,&State::attn,&State::tmp);}
 else{matrix(tensor(b+"attn_q.weight"),s,count,&State::xb,&State::q);matrix(tensor(b+"attn_k.weight"),s,count,&State::xb,&State::k);matrix(tensor(b+"attn_v.weight"),s,count,&State::xb,&State::v);for(int t=0;t<count;t++)attend(s[t],l,position+base+t);matrix(tensor(b+"attn_output.weight"),s,count,&State::attn,&State::tmp);}
 for(int t=0;t<count;t++){for(int i=0;i<1024;i++)s[t].x[i]+=s[t].tmp[i];norm(s[t].xb.data(),s[t].x.data(),tensor(b+"post_attention_norm.weight"),1024);}matrix(tensor(b+"ffn_gate.weight"),s,count,&State::xb,&State::gate);matrix(tensor(b+"ffn_up.weight"),s,count,&State::xb,&State::up);for(int t=0;t<count;t++)for(int i=0;i<3584;i++)s[t].gate[i]*=sigmoid(s[t].gate[i])*s[t].up[i];matrix(tensor(b+"ffn_down.weight"),s,count,&State::gate,&State::tmp);for(int t=0;t<count;t++)for(int i=0;i<1024;i++)s[t].x[i]+=s[t].tmp[i];if(progress)progress(position+base+count-1,l,"layer");}
 if(output&&base+count==tokens.size()){norm(s[count-1].xb.data(),s[count-1].x.data(),tensor("output_norm.weight"),1024);output_head(s[count-1].xb.data());for(float v:logits)req(std::isfinite(v),"nonfinite logits");}
 }next_position+=tokens.size();return logits;}
std::string Engine::stage_metrics()const{return "matrix_us="+std::to_string(matrix_us)+" recurrent_attention_us="+std::to_string(attention_us);}
size_t Engine::validate_kernels()const{
 size_t checked=0;
 for(const auto&t:tensors){if(!t.data||!t.type)continue;
  std::vector<float>a(t.cols),b(t.cols),input(t.cols);dequantize(t.data,t.type,a.data(),t.cols);dequantize_scalar(t.data,t.type,b.data(),t.cols);req(a==b,"dequant kernel parity");
  std::vector<ActivationBlock>packed(t.cols/32*8);for(int lane=0;lane<8;lane++){for(unsigned i=0;i<t.cols;i++)input[i]=float(int((i*17+lane*13)%61)-30)/32;quantize_activation(input.data(),packed.data()+lane,t.cols,8);}float result[8];(t.type==12?q4_activation_dot:q6_activation_dot)(t.data,packed.data(),result,t.cols,8);
#ifdef __ARM_NEON
  std::vector<Q4InputTile>tiles(t.cols/32);for(size_t g=0;g<tiles.size();g++)for(int lane=0;lane<8;lane++){const auto&a=packed[g*8+lane];tiles[g].scale[lane]=a.scale;tiles[g].sum[lane]=a.sum;for(int k=0;k<32;k++)tiles[g].lanes[k][lane]=a.values[k];}float tiled[8];(t.type==12?q4_dot_lane8:q6_dot_lane8)(t.data,tiles.data(),tiled,t.cols);for(int lane=0;lane<8;lane++)req(std::abs(result[lane]-tiled[lane])<=1e-4f*(1+std::abs(result[lane])),"tiled int8 kernel parity");
#endif
  checked+=8;
 }return checked;
}
}
