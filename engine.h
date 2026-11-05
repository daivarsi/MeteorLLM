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
struct ActivationBlock {
 int8_t values[32];
 float scale;
 int sum;
};
void quantize_activation(const float* input,ActivationBlock* blocks,int count,int batch);
void q4_activation_dot(const void* weights,const ActivationBlock* input,float* out,int count,int batch);
void q6_activation_dot(const void* weights,const ActivationBlock* input,float* out,int count,int batch);
#ifdef __ARM_NEON
struct Q4InputTile {
 int8_t lanes[32][8];
 float scale[8],sum[8];
};
void q4_dot_lane8(const void*,const Q4InputTile*,float*,int);
#endif
struct Tensor { std::string name; uint32_t type=0,cols=0,rows=0; uint64_t offset=0,bytes=0; const uint8_t* data=nullptr; };
struct Config { int dim=0,ff=0,layers=0,heads=0,kv_heads=0,head=0,vocab=0,context=0,eos=0;float eps=0,theta=0; };
class Engine {
public:
    // Keep every integer position exactly representable by the float RoPE path.
    static constexpr int position_limit=1<<24;
    Engine()=default;~Engine();
    Engine(const Engine&)=delete;Engine& operator=(const Engine&)=delete;
    void load(const std::string& directory);
    // Reset before reusing the engine after an interrupted forward pass.
    void reset();
    void reset_metrics();
    void save_prefix();
    void restore_prefix();
    size_t kv_bytes() const{return (keys.capacity()+values.capacity())*sizeof(uint16_t);}
    size_t validate_quant_kernels() const;
    void set_workers(int n);
    void set_batch_kernel(int n);
    void set_parallel_attention(bool enabled){parallel_attention=enabled;}
    void set_output_rows(int n);
    bool fast_output=true;
    int output_full_scans=0;
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
    FILE* emb_file=nullptr;
    FILE* q4_output_file=nullptr;
    Tensor emb;
    std::vector<Tensor> tensors;
    std::vector<uint16_t> keys,values,prefix_keys,prefix_values;
    int prefix_length=0;
    std::vector<float> x,xb,tmp,q,k,v,attention,gate,up,scores,logits,row;
    std::vector<uint8_t> io;
    std::vector<ActivationBlock> decode_activation;
    int next_position=0;
};
}
