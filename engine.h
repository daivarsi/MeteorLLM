#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include "matrix_pool.h"
namespace vitaqwen {
inline float half_to_float(uint16_t h){uint32_t sign=uint32_t(h&0x8000)<<16,ex=(h>>10)&31,m=h&1023,b;if(!ex){if(!m)b=sign;else{int e=113;while(!(m&1024)){m<<=1;--e;}b=sign|(uint32_t(e)<<23)|((m&1023)<<13);}}else if(ex==31)b=sign|0x7f800000|(m<<13);else b=sign|((ex+112)<<23)|(m<<13);float f;std::memcpy(&f,&b,4);return f;}
uint16_t float_to_half(float);
void dequantize_scalar(const void*,uint32_t,float*,int);
void dequantize(const void*,uint32_t,float*,int);
float quantized_dot(const void*,uint32_t,const float*,int);
void quantized_dot_batch(const void*,uint32_t,const float*const*,float*,int,int);
struct ActivationBlock{int8_t values[32];float scale;int sum;};
void quantize_activation(const float*,ActivationBlock*,int,int);
void q4_activation_dot(const void*,const ActivationBlock*,float*,int,int);
void q6_activation_dot(const void*,const ActivationBlock*,float*,int,int);
#ifdef __ARM_NEON
struct Q4InputTile{int8_t lanes[32][8];float scale[8],sum[8];};
void q4_dot_lane8(const void*,const Q4InputTile*,float*,int);
void q6_dot_lane8(const void*,const Q4InputTile*,float*,int);
#endif
struct Tensor{std::string name;uint32_t type,cols,rows;uint64_t offset,bytes;const uint8_t*data;};
struct Config{int dim=1024,ff=3584,layers=24,heads=8,kv_heads=2,head=256,vocab=248320,context=512,eos=248046;float eps=1e-6f,theta=1e7f;};
class Engine{
public:
 static constexpr int position_limit=1<<24;
 std::function<bool()>cancelled;uint64_t embedding_read_bytes=0;
 ~Engine();void load(const std::string&);void reset();void save_prefix();void restore_prefix();size_t validate_kernels()const;
 void set_workers(int n){worker_count=n;}void set_parallel_attention(bool){}void set_batch_kernel(int){}void set_output_rows(int n){output_rows=n;}
 const Config&config()const{return cfg;}
 const std::vector<float>&forward(int token,int position,bool output=true){return forward_batch({token},position,output);}
 const std::vector<float>&forward_batch(const std::vector<int>&,int,bool=true);
 uint64_t resident_bytes()const{return model_bytes;}size_t kv_bytes()const{return(keys.size()+values.size())*2;}
 size_t state_bytes()const{return(state.size()+conv.size())*4;}
 void reset_metrics(){matrix_us=attention_us=output_read_us=output_compute_us=0;output_full_scans=0;}
 std::string stage_metrics()const;
 bool fast_output=true,int8=true,q4_head=true,cache_output=true;int output_full_scans=0,shortlist=65536;
 uint64_t matrix_us=0,attention_us=0,output_read_us=0,output_compute_us=0;
 std::function<void(int,int,const char*)>progress;
private:
 struct State{std::vector<float>x,xb,tmp,q,k,v,z,alpha,beta,attn,gate,up;State();};
 Config cfg;MatrixPool pool;int worker_count=3,output_rows=256,next_position=0,prefix_length=0;
 std::string directory;uint8_t*model=nullptr;uint64_t model_bytes=0;int mem_uid=-1;FILE*emb_file=nullptr,*head_file=nullptr;
 std::vector<Tensor>tensors;Tensor emb{};std::vector<float>state,conv,logits;std::vector<uint16_t>keys,values;
 uint8_t*head_cache=nullptr;static constexpr int cached_rows=61440;
 std::vector<uint8_t>io;std::vector<float>scratch[3];
 const Tensor&tensor(const std::string&)const;void norm(float*,const float*,const Tensor&,int);
 void matrix(const Tensor&,State*,int,std::vector<float>State::*,std::vector<float>State::*);
 void embedding(int,float*);void output_head(const float*);void linear(State&,int);void attend(State&,int,int);
};
}
