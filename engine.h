#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include "matrix_pool.h"

namespace vitaqwen {
inline float half_to_float(uint16_t h){
 uint32_t sign=uint32_t(h&0x8000)<<16,ex=(h>>10)&31,m=h&1023,b;
 if(!ex){if(!m)b=sign;else{int e=113;while(!(m&1024)){m<<=1;--e;}b=sign|(uint32_t(e)<<23)|((m&1023)<<13);}}
 else if(ex==31)b=sign|0x7f800000|(m<<13);else b=sign|((ex+112)<<23)|(m<<13);
 float f;std::memcpy(&f,&b,4);return f;
}
uint16_t float_to_half(float value);
void dequantize_scalar(const void *data, uint32_t type, float *out, int count);
void dequantize(const void *data, uint32_t type, float *out, int count);
float quantized_dot(const void *data,uint32_t type,const float *input,int count);
void quantized_dot_batch(const void*,uint32_t,const float* const*,float*,int,int);
struct Tensor { std::string name; uint32_t type=0,cols=0,rows=0; uint64_t offset=0,bytes=0; const uint8_t* data=nullptr; };
struct Config { int dim=0,ff=0,layers=0,heads=0,kv_heads=0,head=0,vocab=0,context=0,eos=0;float eps=0,theta=0; };
class Engine {
public:
    Engine()=default;~Engine();
    Engine(const Engine&)=delete;Engine& operator=(const Engine&)=delete;
    void load(const std::string& directory);
    void reset();
    size_t validate_quant_kernels() const;
    void set_workers(int n);
    void set_batch_kernel(int n);
    void set_parallel_attention(bool enabled){parallel_attention=enabled;}
    void set_output_rows(int n);
    void benchmark_extended(const std::function<void(const std::string&)>& report);
    void validate_extended(const std::function<void(const std::string&)>& report);
    void benchmark_output(const std::function<void(const std::string&)>& report);
    std::string stage_metrics()const;
    uint64_t matrix_us=0,norm_us=0,rope_us=0,attention_us=0,activation_us=0;

    const std::vector<float>& forward_batch(const std::vector<int>& tokens,int position,bool output=true);
    void validate_batch(const std::function<void(const std::string&)>& report);
    void benchmark_matrices(const std::function<void(const std::string&)>& report);
    uint64_t output_read_us=0,output_compute_us=0;
    const std::vector<float>& forward(int token,int position,bool output=true);
    const Config& config() const{return cfg;}
    uint64_t resident_bytes()const{return model_bytes;}
    uint64_t embedding_read_bytes=0;
    std::function<bool()> cancelled;
    std::function<void(int,int,const char*)> progress;
private:
    const Tensor& tensor(const std::string& name)const;
    void matvec(const Tensor&,const float*,float*);
    void norm(float*,const float*,const Tensor&,int);
    void embedding(int token,float*);
    void output_head(const float*);
    void rope(float*,int heads,int position);
    void attend(const float* query,float* result,int layer,int position);
    Config cfg;
    MatrixPool pool;
    int worker_count=1,batch_kernel=0,output_rows=64;
    std::vector<float> scratch[3],attention_scores[3];
    bool parallel_attention=false;
    uint8_t* model=nullptr;uint64_t model_bytes=0;int mem_uid=-1;
    FILE* emb_file=nullptr;Tensor emb;
    std::vector<Tensor> tensors;
    std::vector<uint16_t> keys,values;
    std::vector<float> x,xb,tmp,q,k,v,attention,gate,up,scores,logits,row;
    std::vector<uint8_t> io;
    int next_position=0;
};
}
