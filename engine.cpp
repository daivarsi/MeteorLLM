#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <stdexcept>
#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif
#ifdef __vita__
#include <psp2/kernel/sysmem.h>
#endif
namespace vitaqwen {
static uint64_t now_us(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void require(bool ok,const char*msg){if(!ok)throw std::runtime_error(msg);}
static uint32_t u32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
static uint64_t u64(const uint8_t*p){return u32(p)|(uint64_t(u32(p+4))<<32);}
static float f32(const uint8_t*p){uint32_t b=u32(p);float f;std::memcpy(&f,&b,4);return f;}
static uint64_t tensor_bytes(uint32_t type,uint32_t cols,uint32_t rows){
 uint64_t n=uint64_t(cols)*rows;require(cols&&rows,"zero tensor shape");
 if(type==0)return n*4;
 require(cols%256==0,"bad quant columns");require(type==12||type==14,"unsupported quantization");return n/256*(type==12?144:210);
}
static float dot(const float*a,const float*b,int n){
#ifdef __ARM_NEON
 float32x4_t s=vdupq_n_f32(0);int i=0;for(;i+4<=n;i+=4)s=vmlaq_f32(s,vld1q_f32(a+i),vld1q_f32(b+i));float32x2_t t=vadd_f32(vget_low_f32(s),vget_high_f32(s));t=vpadd_f32(t,t);float v=vget_lane_f32(t,0);for(;i<n;i++)v+=a[i]*b[i];return v;
#elif defined(__SSE2__) || defined(_M_X64)
 __m128 s=_mm_setzero_ps();int i=0;for(;i+4<=n;i+=4)s=_mm_add_ps(s,_mm_mul_ps(_mm_loadu_ps(a+i),_mm_loadu_ps(b+i)));float lane[4];_mm_storeu_ps(lane,s);float v=(lane[0]+lane[2])+(lane[1]+lane[3]);for(;i<n;i++)v+=a[i]*b[i];return v;
#else
 float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;
#endif
}
Engine::~Engine(){if(q4_output_file)std::fclose(q4_output_file);if(emb_file)std::fclose(emb_file);
#ifdef __vita__
 if(mem_uid>=0)sceKernelFreeMemBlock(mem_uid);
#else
 std::free(model);
#endif
}
const Tensor& Engine::tensor(const std::string&name)const{
 for(const auto&t:tensors)if(t.name==name)return t;
 throw std::runtime_error("missing tensor: "+name);
}
void Engine::load(const std::string&dir){
 require(!model,"engine already loaded");FILE*f=std::fopen((dir+"/weights.bin").c_str(),"rb");require(f,"weights open failed");
 if(std::fseek(f,0,SEEK_END)){std::fclose(f);throw std::runtime_error("weights seek failed");}
 long size=std::ftell(f);std::rewind(f);uint8_t h[128];
 if(size<128||size>400L*1024*1024||std::fread(h,1,128,f)!=128){std::fclose(f);throw std::runtime_error("weights size/header failed");}
 try{
 require(std::memcmp(h,"VQWN0001",8)==0,"bad model magic");
 int*fields[]={&cfg.dim,&cfg.ff,&cfg.layers,&cfg.heads,&cfg.kv_heads,&cfg.head,&cfg.vocab,&cfg.context};
 for(int i=0;i<8;i++){*fields[i]=int(u32(h+8+4*i));}
 cfg.eps=f32(h+40);cfg.theta=f32(h+44);uint32_t nt=u32(h+48),payload=u32(h+52);cfg.eos=int(u32(h+56));model_bytes=u64(h+60);
 require(cfg.dim>=256&&cfg.dim<=2048&&cfg.dim%256==0&&cfg.ff>=256&&cfg.ff<=8192&&cfg.ff%256==0,"bad dimensions");
 require(cfg.layers>0&&cfg.layers<=32&&cfg.heads>0&&cfg.heads<=32&&cfg.kv_heads>0&&cfg.heads%cfg.kv_heads==0&&cfg.head>=32&&cfg.head<=256&&cfg.head%2==0,"bad attention dimensions");
 require(cfg.heads*cfg.head<=4096&&cfg.kv_heads*cfg.head<=2048&&cfg.vocab>256&&cfg.vocab<=200000&&cfg.context>0&&cfg.context<=1024&&cfg.eos>=0&&cfg.eos<cfg.vocab,"bad context/vocabulary");
 require(std::isfinite(cfg.eps)&&cfg.eps>=1e-8f&&cfg.eps<=1e-3f&&std::isfinite(cfg.theta)&&cfg.theta>=1&&cfg.theta<=1e7f,"bad norm/rope config");
 require(nt==uint32_t(11*cfg.layers+2)&&payload==((128+nt*96+255)/256)*256&&payload<=model_bytes&&model_bytes==uint64_t(size),"bad model table bounds");
 for(int i=68;i<128;i++)require(h[i]==0,"reserved header bytes");
#ifdef __vita__
 mem_uid=sceKernelAllocMemBlock("VitaQwenWeights",SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,(size+4095)&~4095,nullptr);require(mem_uid>=0,"model memblock allocation failed");void*p=nullptr;require(sceKernelGetMemBlockBase(mem_uid,&p)>=0,"model memblock mapping failed");model=static_cast<uint8_t*>(p);
#else
 model=static_cast<uint8_t*>(std::malloc(size));require(model,"model allocation failed");
#endif
 std::rewind(f);require(std::fread(model,1,size,f)==size_t(size),"weights short read");std::fclose(f);f=nullptr;
 uint64_t cursor=payload;int external=0;
 for(uint32_t i=0;i<nt;i++){
  const uint8_t*p=model+128+i*96;require(std::memchr(p,0,64)!=nullptr,"unterminated tensor name");Tensor t;t.name=reinterpret_cast<const char*>(p);t.type=u32(p+64);t.cols=u32(p+68);t.rows=u32(p+72);t.offset=u64(p+76);t.bytes=u64(p+84);
  require(u32(p+92)==0&&t.cols<=8192&&t.rows<=200000,"bad tensor metadata");require(t.bytes==tensor_bytes(t.type,t.cols,t.rows),"tensor size mismatch");
  for(const auto&other:tensors)require(t.name!=other.name,"duplicate tensor");
  if(t.offset==UINT64_MAX){require(t.name=="token_embd.weight","unexpected external tensor");external++;emb=t;}
  else{require(t.offset==cursor&&cursor<=model_bytes&&t.bytes<=model_bytes-cursor,"tensor payload out of bounds");t.data=model+cursor;cursor+=t.bytes;}
  tensors.push_back(t);
 }
 require(cursor==model_bytes&&external==1,"incomplete tensor layout");
 auto shape=[&](const std::string&name,int cols,int rows,bool norm=false){const auto&t=tensor(name);require(t.cols==uint32_t(cols)&&t.rows==uint32_t(rows)&&(!norm||t.type==0),"unexpected tensor shape/type");};
 shape("token_embd.weight",cfg.dim,cfg.vocab);shape("output_norm.weight",cfg.dim,1,true);
 for(int l=0;l<cfg.layers;l++){std::string b="blk."+std::to_string(l)+".";
  shape(b+"attn_norm.weight",cfg.dim,1,true);shape(b+"ffn_norm.weight",cfg.dim,1,true);
  shape(b+"attn_q_norm.weight",cfg.head,1,true);shape(b+"attn_k_norm.weight",cfg.head,1,true);
  shape(b+"attn_q.weight",cfg.dim,cfg.heads*cfg.head);shape(b+"attn_k.weight",cfg.dim,cfg.kv_heads*cfg.head);shape(b+"attn_v.weight",cfg.dim,cfg.kv_heads*cfg.head);shape(b+"attn_output.weight",cfg.heads*cfg.head,cfg.dim);
  shape(b+"ffn_gate.weight",cfg.dim,cfg.ff);shape(b+"ffn_up.weight",cfg.dim,cfg.ff);shape(b+"ffn_down.weight",cfg.ff,cfg.dim);
 }
 emb_file=std::fopen((dir+"/embedding.bin").c_str(),"rb");require(emb_file,"embedding open failed");require(std::fseek(emb_file,0,SEEK_END)==0&&uint64_t(std::ftell(emb_file))==emb.bytes,"embedding size mismatch");
 q4_output_file=std::fopen((dir+"/output-q4-test.bin").c_str(),"rb");
 if(q4_output_file){
  const uint64_t expected=uint64_t(cfg.vocab)*(cfg.dim/256)*144;
  require(std::fseek(q4_output_file,0,SEEK_END)==0&&uint64_t(std::ftell(q4_output_file))==expected,"Q4 output size mismatch");
 }
 int qdim=cfg.heads*cfg.head,kdim=cfg.kv_heads*cfg.head;
 size_t nk=size_t(cfg.layers)*cfg.context*kdim;require(nk<=64*1024*1024,"KV allocation bound");keys.resize(nk);values.resize(nk);
 x.resize(cfg.dim);xb.resize(cfg.dim);tmp.resize(cfg.dim);q.resize(qdim);k.resize(kdim);v.resize(kdim);attention.resize(qdim);gate.resize(cfg.ff);up.resize(cfg.ff);scores.resize(cfg.context);logits.resize(cfg.vocab);row.resize(std::max({cfg.dim,cfg.ff,qdim}));io.resize(size_t(emb.bytes/emb.rows)*64);for(auto& s:scratch)s.resize(row.size());for(auto& s:attention_scores)s.resize(cfg.context);reset();
 }catch(...){if(f)std::fclose(f);throw;}
}
size_t Engine::validate_quant_kernels()const{
 size_t checked=0;float actual[256],expected[256],input[256];for(int i=0;i<256;i++)input[i]=float(i%31-15)/32;
 for(const auto&t:tensors){if(!t.data||(t.type!=12&&t.type!=14))continue;size_t block=t.type==12?144:210,blocks=t.bytes/block;
  for(size_t sample=0;sample<8;sample++){size_t offset=(sample*(blocks-1)/7)*block;
   dequantize(t.data+offset,t.type,actual,256);dequantize_scalar(t.data+offset,t.type,expected,256);
   require(std::memcmp(actual,expected,sizeof(actual))==0,"quant kernel parity failed");
   float ref=dot(expected,input,256),fast=quantized_dot(t.data+offset,t.type,input,256);
   require(std::memcmp(&ref,&fast,sizeof(float))==0,"quant dot parity failed");checked+=256;
  }
  std::vector<float> decoded(t.cols),inputs(t.cols);uint32_t seed=734;
  for(size_t sample=0;sample<2;sample++){
   for(auto&value:inputs){seed=1664525u*seed+1013904223u;value=float(int(seed>>16)-32768)/32768;}
   const uint8_t*ptr=t.data+(sample?(t.rows-1)*(t.bytes/t.rows):0);
   dequantize_scalar(ptr,t.type,decoded.data(),t.cols);
   float ref=dot(decoded.data(),inputs.data(),t.cols),fast=quantized_dot(ptr,t.type,inputs.data(),t.cols);
   require(std::memcmp(&ref,&fast,sizeof(float))==0,"full-row quant dot parity failed");checked+=t.cols;
  }
 }return checked;
}
void Engine::reset(){
 // A new sequence only attends to slots written since this reset.
 next_position=0;reset_metrics();
}
void Engine::save_prefix(){
 require(next_position>0&&next_position<=cfg.context,"invalid prefix length");
 prefix_length=next_position;
 const size_t width=cfg.kv_heads*cfg.head;
 const size_t count=size_t(prefix_length)*width;
 prefix_keys.resize(cfg.layers*count);
 prefix_values.resize(cfg.layers*count);
 for(int layer=0;layer<cfg.layers;layer++){
  const size_t source=size_t(layer)*cfg.context*width;
  std::copy_n(keys.data()+source,count,prefix_keys.data()+layer*count);
  std::copy_n(values.data()+source,count,prefix_values.data()+layer*count);
 }
}
void Engine::restore_prefix(){
 require(prefix_length>0,"missing prefix");
 const size_t width=cfg.kv_heads*cfg.head;
 const size_t count=size_t(prefix_length)*width;
 for(int layer=0;layer<cfg.layers;layer++){
  const size_t destination=size_t(layer)*cfg.context*width;
  std::copy_n(prefix_keys.data()+layer*count,count,keys.data()+destination);
  std::copy_n(prefix_values.data()+layer*count,count,values.data()+destination);
 }
 next_position=prefix_length;
}
void Engine::reset_metrics(){output_full_scans=0;embedding_read_bytes=0;output_read_us=output_compute_us=0;matrix_us=norm_us=rope_us=attention_us=activation_us=0;}
void Engine::set_batch_kernel(int n){require(n>=0&&n<=2,"invalid batch kernel");batch_kernel=n;}
void Engine::set_output_rows(int n){require(n>=64&&n<=1024,"invalid output chunk size");output_rows=n;if(emb.rows)io.resize(size_t(emb.bytes/emb.rows)*n);}
std::string Engine::stage_metrics()const{return "matrix_us="+std::to_string(matrix_us)+" norm_us="+std::to_string(norm_us)+" rope_us="+std::to_string(rope_us)+" attention_us="+std::to_string(attention_us)+" activation_us="+std::to_string(activation_us);}
void Engine::set_workers(int n){require(n>=1&&n<=3,"invalid worker count");worker_count=n;}
void Engine::matvec(const Tensor&t,const float*in,float*out){
 auto stage_start=now_us();require(t.data,"resident matvec expected");size_t stride=t.bytes/t.rows;
 if(cancelled&&cancelled())throw std::runtime_error("Stopped");
 const bool use_int8=t.type==12||t.type==14;
 const auto integer_dot=t.type==12?q4_activation_dot:q6_activation_dot;
 if(use_int8){
  decode_activation.resize(t.cols/32);
  quantize_activation(in,decode_activation.data(),t.cols,1);
 }
 pool.run(worker_count,t.rows,[&](int begin,int end,[[maybe_unused]] int worker){
  if(use_int8){
   for(int r=begin;r<end;r++)
    integer_dot(t.data+r*stride,decode_activation.data(),out+r,t.cols,1);
   return;
  }
  for(int r=begin;r<end;r++){
#ifdef __ARM_NEON
   out[r]=quantized_dot(t.data+r*stride,t.type,in,t.cols);
#else
   dequantize(t.data+r*stride,t.type,scratch[worker].data(),t.cols);out[r]=dot(scratch[worker].data(),in,t.cols);
#endif
  }
 });
 matrix_us+=now_us()-stage_start;if(cancelled&&cancelled())throw std::runtime_error("Stopped");
}
void Engine::benchmark_matrices(const std::function<void(const std::string&)>& report){
 const int saved=worker_count;
 try{
  for(const auto& t:tensors){if(!t.data||(t.type!=12&&t.type!=14))continue;
   // All shapes/types, all rows, with deterministic representative inputs.
   if(t.name.rfind("blk.0.",0)!=0&&t.name.rfind("blk.27.",0)!=0)continue;
   std::vector<float> in(t.cols),ref(t.rows),out(t.rows);uint32_t seed=734;
   for(auto& value:in){seed=1664525u*seed+1013904223u;value=float(int(seed>>16)-32768)/32768;}
   set_workers(1);matvec(t,in.data(),ref.data());
   for(int n=1;n<=3;n++){set_workers(n);uint64_t elapsed=0;
    for(int repeat=0;repeat<3;repeat++){auto start=now_us();matvec(t,in.data(),out.data());elapsed+=now_us()-start;require(std::memcmp(ref.data(),out.data(),t.rows*sizeof(float))==0,"parallel matrix parity failed");}
    report("tensor="+t.name+" workers="+std::to_string(n)+" repeats=3 elapsed_us="+std::to_string(elapsed)+" rows="+std::to_string(t.rows)+" bit_exact=1");
   }
  }
 }catch(...){worker_count=saved;throw;}worker_count=saved;
}
void Engine::norm(float*out,const float*in,const Tensor&w,int n){
 auto stage_start=now_us();float sum=dot(in,in,n),scale=1/std::sqrt(sum/n+cfg.eps);const float*weights=reinterpret_cast<const float*>(w.data);
 for(int i=0;i<n;i++){out[i]=in[i]*scale*weights[i];}
 norm_us+=now_us()-stage_start;
}
void Engine::embedding(int token,float*out){size_t stride=emb.bytes/emb.rows;require(std::fseek(emb_file,long(uint64_t(token)*stride),SEEK_SET)==0,"embedding seek failed");require(std::fread(io.data(),1,stride,emb_file)==stride,"embedding short read");embedding_read_bytes+=stride;dequantize(io.data(),emb.type,out,cfg.dim);}
void Engine::output_head(const float*in){
 FILE* file=q4_output_file?q4_output_file:emb_file;
 const size_t stride=q4_output_file?size_t(cfg.dim/256)*144:emb.bytes/emb.rows;
 require(std::fseek(file,0,SEEK_SET)==0,"output seek failed");
 if(q4_output_file){
  decode_activation.resize(cfg.dim/32);
  quantize_activation(in,decode_activation.data(),cfg.dim,1);
 }
 const auto project=[&](int first,int end){
 require(std::fseek(file,long(size_t(first)*stride),SEEK_SET)==0,"output range seek failed");
 for(int base=first;base<end;base+=output_rows){if(cancelled&&cancelled())throw std::runtime_error("Stopped");int nr=std::min(output_rows,end-base);size_t bytes=stride*nr;auto read_start=now_us();require(std::fread(io.data(),1,bytes,file)==bytes,"output short read");output_read_us+=now_us()-read_start;embedding_read_bytes+=bytes;auto compute_start=now_us();
  pool.run(worker_count,nr,[&](int begin,int end,[[maybe_unused]] int worker){for(int r=begin;r<end;r++){
 if(q4_output_file){
  q4_activation_dot(io.data()+r*stride,decode_activation.data(),logits.data()+base+r,cfg.dim,1);
  continue;
 }
#ifdef __ARM_NEON
 logits[base+r]=quantized_dot(io.data()+r*stride,emb.type,in,cfg.dim);
#else
 dequantize(io.data()+r*stride,emb.type,scratch[worker].data(),cfg.dim);logits[base+r]=dot(scratch[worker].data(),in,cfg.dim);
#endif
 }});output_compute_us+=now_us()-compute_start;
 }
 };
 // This fast path is specific to the Qwen3 vocabulary layout.
 constexpr int prefix=24576, tail=151643;
 if(!fast_output||!q4_output_file||cfg.vocab!=151936||cfg.eos!=151645){
  project(0,cfg.vocab);
  ++output_full_scans;
  return;
 }
 std::fill(logits.begin()+prefix,logits.begin()+tail,-1e30f);
 project(0,prefix);
 project(tail,cfg.vocab);
 float top=-1e30f,runner_up=-1e30f;
 for(float score:logits){
  if(score>top){runner_up=top;top=score;}
  else if(score>runner_up)runner_up=score;
 }
 if(top-runner_up<1.f){
  project(prefix,tail);
  ++output_full_scans;
 }

}
void Engine::attend(const float*query,float*result,int layer,int position){
 const int kd=cfg.kv_heads*cfg.head;
 const size_t cachebase=size_t(layer)*cfg.context*kd;
 // Visit the ring oldest first, including across the physical wrap.
 const int count=std::min(position+1,cfg.context);
 const int oldest=position<cfg.context?0:(position+1)%cfg.context;
 const int first_count=std::min(count,cfg.context-oldest);
 std::atomic<bool> valid{true};
 pool.run(parallel_attention?worker_count:1,cfg.heads,[&](int first,int last,int worker){auto&scores=attention_scores[worker];
  for(int h=first;h<last;h++){
   int kh=h/(cfg.heads/cfg.kv_heads);float mx=-std::numeric_limits<float>::infinity();
   for(int p=0,slot=oldest;p<count;p++,slot++){if(slot==cfg.context)slot=0;float sum=0;size_t at=cachebase+size_t(slot)*kd+kh*cfg.head;for(int i=0;i<cfg.head;i++)sum+=query[h*cfg.head+i]*half_to_float(keys[at+i]);scores[p]=sum/std::sqrt(float(cfg.head));mx=std::max(mx,scores[p]);}
   float sum=0;for(int p=0;p<count;p++){scores[p]=std::exp(scores[p]-mx);sum+=scores[p];}if(!std::isfinite(sum)||sum<=0){valid.store(false,std::memory_order_relaxed);continue;}for(int p=0;p<count;p++)scores[p]/=sum;
   for(int i=0;i<cfg.head;i++){float a=0;const auto* value=values.data()+cachebase+kh*cfg.head+i;for(int p=0;p<first_count;p++)a+=scores[p]*half_to_float(value[size_t(oldest+p)*kd]);for(int p=first_count;p<count;p++)a+=scores[p]*half_to_float(value[size_t(p-first_count)*kd]);result[h*cfg.head+i]=a;}
  }
 });
 require(valid.load(std::memory_order_relaxed),"invalid attention scores");
}
void Engine::rope(float*data,int heads,int position){auto stage_start=now_us();
 for(int h=0;h<heads;h++)for(int i=0;i<cfg.head/2;i++){float angle=position/std::pow(cfg.theta,2.0f*i/cfg.head),c=std::cos(angle),s=std::sin(angle);int a=h*cfg.head+i,b=a+cfg.head/2;float va=data[a],vb=data[b];data[a]=va*c-vb*s;data[b]=va*s+vb*c;}
 rope_us+=now_us()-stage_start;
}
const std::vector<float>& Engine::forward(int token,int position,bool output){
 if(progress)progress(position,-1,"start");
 require(model&&token>=0&&token<cfg.vocab&&position==next_position&&position<position_limit,"invalid forward position/token");embedding(token,x.data());int kd=cfg.kv_heads*cfg.head;
 for(int l=0;l<cfg.layers;l++){
  if(cancelled&&cancelled())throw std::runtime_error("Stopped");
  std::string b="blk."+std::to_string(l)+".";norm(xb.data(),x.data(),tensor(b+"attn_norm.weight"),cfg.dim);
  matvec(tensor(b+"attn_q.weight"),xb.data(),q.data());matvec(tensor(b+"attn_k.weight"),xb.data(),k.data());matvec(tensor(b+"attn_v.weight"),xb.data(),v.data());
  if(progress&&position==0&&l==0)progress(position,l,"qkv");
  for(int h=0;h<cfg.heads;h++)norm(q.data()+h*cfg.head,q.data()+h*cfg.head,tensor(b+"attn_q_norm.weight"),cfg.head);
  for(int h=0;h<cfg.kv_heads;h++)norm(k.data()+h*cfg.head,k.data()+h*cfg.head,tensor(b+"attn_k_norm.weight"),cfg.head);
  rope(q.data(),cfg.heads,position);rope(k.data(),cfg.kv_heads,position);
  auto attention_start=now_us();size_t cachebase=size_t(l)*cfg.context*kd;
  const size_t slot=cachebase+size_t(position%cfg.context)*kd;
  for(int i=0;i<kd;i++){keys[slot+i]=float_to_half(k[i]);values[slot+i]=float_to_half(v[i]);}
  attend(q.data(),attention.data(),l,position);
  attention_us+=now_us()-attention_start;
  if(progress&&position==0&&l==0)progress(position,l,"attention");
  matvec(tensor(b+"attn_output.weight"),attention.data(),tmp.data());for(int i=0;i<cfg.dim;i++)x[i]+=tmp[i];norm(xb.data(),x.data(),tensor(b+"ffn_norm.weight"),cfg.dim);
  matvec(tensor(b+"ffn_gate.weight"),xb.data(),gate.data());matvec(tensor(b+"ffn_up.weight"),xb.data(),up.data());for(int i=0;i<cfg.ff;i++)gate[i]=gate[i]/(1+std::exp(-gate[i]))*up[i];matvec(tensor(b+"ffn_down.weight"),gate.data(),tmp.data());for(int i=0;i<cfg.dim;i++)x[i]+=tmp[i];
  if(progress)progress(position,l,"layer");
 }
 ++next_position;if(output){norm(xb.data(),x.data(),tensor("output_norm.weight"),cfg.dim);output_head(xb.data());for(float value:logits)require(std::isfinite(value),"nonfinite logits");}return logits;
}

const std::vector<float>& Engine::forward_batch(const std::vector<int>& tokens,int position,bool output){
 require(!tokens.empty()&&tokens.size()<=16,"batch size must be 1..16");
 require(model&&position==next_position&&position>=0&&position<=position_limit-int(tokens.size()),"invalid batch position");
 for(int token:tokens)require(token>=0&&token<cfg.vocab,"invalid batch token");
 if(tokens.size()==1)return forward(tokens[0],position,output);
 struct State{std::vector<float>x,xb,tmp,q,k,v,attention,gate,up;};
 State states[16];int count=int(tokens.size()),kd=cfg.kv_heads*cfg.head,qd=cfg.heads*cfg.head;
 for(int t=0;t<count;t++){auto&s=states[t];s.x.resize(cfg.dim);s.xb.resize(cfg.dim);s.tmp.resize(cfg.dim);s.q.resize(qd);s.k.resize(kd);s.v.resize(kd);s.attention.resize(qd);s.gate.resize(cfg.ff);s.up.resize(cfg.ff);embedding(tokens[t],s.x.data());}
 auto matrix=[&](const Tensor&w,std::vector<float> State::*input,std::vector<float> State::*output){
  if(cancelled&&cancelled())throw std::runtime_error("Stopped");
  auto stage_start=now_us();size_t stride=w.bytes/w.rows;const float* inputs[16];for(int t=0;t<count;t++)inputs[t]=(states[t].*input).data();
  std::vector<ActivationBlock> packed;
  if(w.type==12||w.type==14){
   packed.resize(size_t(w.cols/32)*count);
   for(int t=0;t<count;t++)quantize_activation(inputs[t],packed.data()+t,w.cols,count);
  }
#ifdef __ARM_NEON
  std::vector<Q4InputTile> tiles;
  const auto lane_dot=w.type==12?q4_dot_lane8:q6_dot_lane8;
  if((w.type==12||w.type==14)&&count==8){
   tiles.resize(w.cols/32);
   for(size_t g=0;g<tiles.size();g++)for(int t=0;t<8;t++){
    const auto& a=packed[g*8+t];
    auto& tile=tiles[g];
    tile.scale[t]=a.scale;
    tile.sum[t]=float(a.sum);
    for(int k=0;k<32;k++)tile.lanes[k][t]=a.values[k];
   }
  }
#endif
  const auto integer_dot=w.type==12?q4_activation_dot:q6_activation_dot;
  pool.run(worker_count,w.rows,[&](int begin,int end,[[maybe_unused]] int worker){
   float* decoded=scratch[worker].data();
   for(int r=begin;r<end;r++){
#ifdef __ARM_NEON
    if(!tiles.empty()){
     float result[8];
     lane_dot(w.data+r*stride,tiles.data(),result,w.cols);
     for(int t=0;t<8;t++)(states[t].*output)[r]=result[t];
     continue;
    }
#endif
    if(!packed.empty()){
     float result[16];
     integer_dot(w.data+r*stride,packed.data(),result,w.cols,count);
     for(int t=0;t<count;t++)(states[t].*output)[r]=result[t];
    }
    else if(batch_kernel){int width=batch_kernel==1?4:8;float results[8];for(int base=0;base<count;base+=width){int n=std::min(width,count-base);quantized_dot_batch(w.data+r*stride,w.type,inputs+base,results,w.cols,n);for(int t=0;t<n;t++)(states[base+t].*output)[r]=results[t];}}
    else{dequantize(w.data+r*stride,w.type,decoded,w.cols);for(int t=0;t<count;t++)(states[t].*output)[r]=dot(decoded,inputs[t],w.cols);}
   }
  });
  matrix_us+=now_us()-stage_start;if(cancelled&&cancelled())throw std::runtime_error("Stopped");
 };
 for(int l=0;l<cfg.layers;l++){
  std::string b="blk."+std::to_string(l)+".";
  for(int t=0;t<count;t++){auto&s=states[t];norm(s.xb.data(),s.x.data(),tensor(b+"attn_norm.weight"),cfg.dim);}
  matrix(tensor(b+"attn_q.weight"),&State::xb,&State::q);matrix(tensor(b+"attn_k.weight"),&State::xb,&State::k);matrix(tensor(b+"attn_v.weight"),&State::xb,&State::v);
  for(int t=0;t<count;t++){
   int position=next_position+t;auto&s=states[t];auto&q=s.q;auto&k=s.k;auto&v=s.v;auto&attention=s.attention;
  for(int h=0;h<cfg.heads;h++)norm(q.data()+h*cfg.head,q.data()+h*cfg.head,tensor(b+"attn_q_norm.weight"),cfg.head);
  for(int h=0;h<cfg.kv_heads;h++)norm(k.data()+h*cfg.head,k.data()+h*cfg.head,tensor(b+"attn_k_norm.weight"),cfg.head);
  rope(q.data(),cfg.heads,position);rope(k.data(),cfg.kv_heads,position);
  auto attention_start=now_us();size_t cachebase=size_t(l)*cfg.context*kd;
  const size_t slot=cachebase+size_t(position%cfg.context)*kd;
  for(int i=0;i<kd;i++){keys[slot+i]=float_to_half(k[i]);values[slot+i]=float_to_half(v[i]);}
  attend(q.data(),attention.data(),l,position);
  attention_us+=now_us()-attention_start;
  }
  matrix(tensor(b+"attn_output.weight"),&State::attention,&State::tmp);
  for(int t=0;t<count;t++){auto&s=states[t];for(int i=0;i<cfg.dim;i++)s.x[i]+=s.tmp[i];norm(s.xb.data(),s.x.data(),tensor(b+"ffn_norm.weight"),cfg.dim);}
  matrix(tensor(b+"ffn_gate.weight"),&State::xb,&State::gate);matrix(tensor(b+"ffn_up.weight"),&State::xb,&State::up);
  auto activation_start=now_us();for(int t=0;t<count;t++){auto&s=states[t];for(int i=0;i<cfg.ff;i++)s.gate[i]=s.gate[i]/(1+std::exp(-s.gate[i]))*s.up[i];}
  activation_us+=now_us()-activation_start;matrix(tensor(b+"ffn_down.weight"),&State::gate,&State::tmp);
  for(int t=0;t<count;t++){auto&s=states[t];for(int i=0;i<cfg.dim;i++)s.x[i]+=s.tmp[i];}
  if(progress)progress(position+count-1,l,"batch_layer");
 }
 next_position+=count;
 if(output){norm(xb.data(),states[count-1].x.data(),tensor("output_norm.weight"),cfg.dim);output_head(xb.data());for(float value:logits)require(std::isfinite(value),"nonfinite batch logits");}
 return logits;
}
void Engine::validate_batch(const std::function<void(const std::string&)>& report){
 int saved=worker_count;auto old_cancel=cancelled;auto old_progress=progress;progress={};cancelled={};
 try{
  // A four-token prefix followed by a decode checks KV contents as well as logits.
  std::vector<int> ids{151644,894,198,2610};
  set_workers(1);reset();for(int i=0;i<4;i++)forward(ids[i],i,i==3);
  auto reference=logits;std::vector<uint16_t> reference_keys,reference_values;
  for(int l=0;l<cfg.layers;l++){size_t at=size_t(l)*cfg.context*cfg.kv_heads*cfg.head,n=4*cfg.kv_heads*cfg.head;reference_keys.insert(reference_keys.end(),keys.begin()+at,keys.begin()+at+n);reference_values.insert(reference_values.end(),values.begin()+at,values.begin()+at+n);}
  // Only initialized prefix slots are compared, not future slots.
  auto compare_cache=[&](){size_t kd=cfg.kv_heads*cfg.head;for(int l=0;l<cfg.layers;l++){size_t at=size_t(l)*cfg.context*kd;require(std::memcmp(keys.data()+at,reference_keys.data()+size_t(l)*4*kd,4*kd*2)==0&&std::memcmp(values.data()+at,reference_values.data()+size_t(l)*4*kd,4*kd*2)==0,"batch KV mismatch");}};
  auto continuation=forward(374,4,true);
  for(int n=1;n<=3;n++){
   set_workers(n);reset();forward_batch(ids,0,true);compare_cache();require(std::memcmp(logits.data(),reference.data(),logits.size()*4)==0,"batch full logits mismatch");
   forward(374,4,true);require(std::memcmp(logits.data(),continuation.data(),logits.size()*4)==0,"batch continuation mismatch");
   report("workers="+std::to_string(n)+" batch=4 full_logits_bit_exact=1 kv_bit_exact=1 continuation_bit_exact=1");
  }
  reset();cancelled=[](){return true;};bool stopped=false;try{forward_batch(ids,0,false);}catch(const std::runtime_error&e){stopped=std::string(e.what())=="Stopped";}require(stopped,"batch cancellation failed");cancelled={};
  reset();forward_batch({ids[0],ids[1]},0,false);forward_batch({ids[2],ids[3]},2,true);require(std::memcmp(logits.data(),reference.data(),logits.size()*4)==0,"split batch logits mismatch");
  report("batch=2 nonzero_position=1 cancellation=1 recovery=1 full_logits_bit_exact=1");
 }catch(...){worker_count=saved;cancelled=old_cancel;progress=old_progress;reset();throw;}
 worker_count=saved;cancelled=old_cancel;progress=old_progress;reset();
}


void Engine::benchmark_extended(const std::function<void(const std::string&)>& report){
 // Real model rows, all resident quantized tensors, widths 1..8, zero,
 // alternating magnitudes and deterministic pseudo-random activation vectors.
 size_t checked=0;
 for(const auto&t:tensors){if(!t.data||(t.type!=12&&t.type!=14))continue;
  std::vector<float> input[8];const float*ptr[8];uint32_t seed=734;
  for(int b=0;b<8;b++){input[b].resize(t.cols);ptr[b]=input[b].data();for(uint32_t i=0;i<t.cols;i++){seed=1664525u*seed+1013904223u;input[b][i]=b==0?0:b==1?(i%2?8.f:-8.f):float(int(seed>>16)-32768)/32768;}}
  for(int sample=0;sample<2;sample++){const uint8_t*row=t.data+(sample?t.rows-1:0)*(t.bytes/t.rows);float ref[8],out[8];for(int b=0;b<8;b++)ref[b]=quantized_dot(row,t.type,ptr[b],t.cols);
   for(int n=1;n<=8;n++){quantized_dot_batch(row,t.type,ptr,out,t.cols,n);require(std::memcmp(ref,out,n*sizeof(float))==0,"fused batch row mismatch");checked+=n;}
  }
 }
 report("row_outputs="+std::to_string(checked)+" widths=1..8 bit_exact=1");
 int saved=worker_count;set_workers(3);
 for(const auto&t:tensors){if(t.name!="blk.0.ffn_gate.weight"&&t.name!="blk.0.ffn_down.weight"&&t.name!="blk.0.attn_q.weight")continue;
  std::vector<float> input[16],out(t.rows*16),reference(t.rows*16);const float*ptr[16];
  for(int b=0;b<16;b++){input[b].resize(t.cols);ptr[b]=input[b].data();for(uint32_t i=0;i<t.cols;i++)input[b][i]=float(int((i*17+b*13)%61)-30)/32;}
  for(int batch:{4,8,16})for(int mode=0;mode<=2;mode++){uint64_t elapsed=0;
   for(int repeat=0;repeat<3;repeat++){auto begin=now_us();pool.run(worker_count,t.rows,[&](int first,int last,int worker){for(int r=first;r<last;r++){
    const uint8_t*data=t.data+r*(t.bytes/t.rows);
    if(mode){int width=mode==1?4:8;float result[8];for(int b=0;b<batch;b+=width){int n=std::min(width,batch-b);quantized_dot_batch(data,t.type,ptr+b,result,t.cols,n);for(int j=0;j<n;j++)out[(b+j)*t.rows+r]=result[j];}}
    else{dequantize(data,t.type,scratch[worker].data(),t.cols);for(int b=0;b<batch;b++)out[b*t.rows+r]=dot(scratch[worker].data(),ptr[b],t.cols);}
   }});elapsed+=now_us()-begin;if(mode==0)reference=out;else require(std::memcmp(reference.data(),out.data(),batch*t.rows*sizeof(float))==0,"fused batch matrix mismatch");}
   report("tensor="+t.name+" batch="+std::to_string(batch)+" kernel="+std::to_string(mode)+" repeats=3 elapsed_us="+std::to_string(elapsed)+" bit_exact=1");
  }
 }set_workers(saved);
}
void Engine::validate_extended(const std::function<void(const std::string&)>& report){
 const int saved_workers=worker_count,saved_kernel=batch_kernel;const bool saved_attention=parallel_attention;auto cancel=cancelled;auto progress_saved=progress;cancelled={};progress={};
 try{
  std::vector<int> ids{151644,894,198,2610,525,264,10950,17847,13,11281,279,11470,323,4226,621,13};
  set_workers(3);set_batch_kernel(0);parallel_attention=false;reset();for(int i=0;i<16;i++)forward(ids[i],i,i==15);
  auto reference=logits;std::vector<uint16_t> key_ref,value_ref;size_t kd=cfg.kv_heads*cfg.head;
  for(int l=0;l<cfg.layers;l++){size_t at=size_t(l)*cfg.context*kd;key_ref.insert(key_ref.end(),keys.begin()+at,keys.begin()+at+16*kd);value_ref.insert(value_ref.end(),values.begin()+at,values.begin()+at+16*kd);}
  auto continuation=forward(198,16,true);
  for(int batch:{8,16}){parallel_attention=true;set_batch_kernel(2);reset();for(int p=0;p<16;p+=batch)forward_batch(std::vector<int>(ids.begin()+p,ids.begin()+p+batch),p,p+batch==16);
   require(std::memcmp(reference.data(),logits.data(),logits.size()*4)==0,"extended logits mismatch");
   for(int l=0;l<cfg.layers;l++){size_t at=size_t(l)*cfg.context*kd,ref=size_t(l)*16*kd;require(std::memcmp(keys.data()+at,key_ref.data()+ref,16*kd*2)==0&&std::memcmp(values.data()+at,value_ref.data()+ref,16*kd*2)==0,"extended KV mismatch");}
   forward(198,16,true);require(std::memcmp(continuation.data(),logits.data(),logits.size()*4)==0,"extended continuation mismatch");report("batch="+std::to_string(batch)+" kernel=2 parallel_attention=1 logits_bit_exact=1 kv_bit_exact=1 continuation_bit_exact=1");
  }
  reset();cancelled=[](){return true;};bool stopped=false;try{forward_batch(ids,0,false);}catch(const std::runtime_error&e){stopped=std::string(e.what())=="Stopped";}require(stopped,"extended cancellation failed");cancelled={};reset();
  bool invalid=false;try{forward_batch(std::vector<int>(17,198),0,false);}catch(const std::runtime_error&){invalid=true;}require(invalid,"oversized batch accepted");
  set_batch_kernel(1);for(int p=0;p<16;p+=8)forward_batch(std::vector<int>(ids.begin()+p,ids.begin()+p+8),p,p==8);require(std::memcmp(reference.data(),logits.data(),logits.size()*4)==0,"kernel1 recovery mismatch");
  report("kernel=1 batch=8 nonzero_position=1 cancellation=1 recovery=1 oversize_rejected=1 logits_bit_exact=1");
 }catch(...){set_workers(saved_workers);set_batch_kernel(saved_kernel);parallel_attention=saved_attention;cancelled=cancel;progress=progress_saved;reset();throw;}
 set_workers(saved_workers);set_batch_kernel(saved_kernel);parallel_attention=saved_attention;cancelled=cancel;progress=progress_saved;reset();
}
void Engine::benchmark_output(const std::function<void(const std::string&)>& report){
 int saved_workers=worker_count,saved_rows=output_rows;set_workers(3);std::vector<float> input(cfg.dim),reference;
 for(int i=0;i<cfg.dim;i++)input[i]=float((i*17)%61-30)/32;
 try{for(int rows:{64,256,1024}){set_output_rows(rows);for(int repeat=0;repeat<2;repeat++){reset();auto start=now_us();output_head(input.data());if(reference.empty())reference=logits;else require(std::memcmp(reference.data(),logits.data(),logits.size()*4)==0,"output chunk logits mismatch");report("rows="+std::to_string(rows)+" repeat="+std::to_string(repeat)+" elapsed_us="+std::to_string(now_us()-start)+" read_us="+std::to_string(output_read_us)+" compute_us="+std::to_string(output_compute_us)+" bit_exact=1");}}}
 catch(...){set_workers(saved_workers);set_output_rows(saved_rows);reset();throw;}set_workers(saved_workers);set_output_rows(saved_rows);reset();
}

}
