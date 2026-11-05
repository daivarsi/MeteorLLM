#include <algorithm>
// Q4_K/Q6_K layouts and dequantization adapted from ggml, pinned at 19e8a86fe.
// Copyright (c) 2023-2026 The ggml authors. MIT license: licenses/llama.cpp-LICENSE.
#include "engine.h"
#include <cstring>
#include <stdexcept>
#include <cmath>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
namespace vitaqwen {
uint16_t float_to_half(float f){
 uint32_t u;std::memcpy(&u,&f,4);uint16_t s=(u>>16)&0x8000;uint32_t exp=(u>>23)&255,m=u&0x7fffff;
 if(exp==255)return s|0x7c00|(m?0x200:0);
 int e=int(exp)-127+15;if(e>=31)return s|0x7c00;
 if(e<=0){if(e < -10)return s;m|=0x800000;int shift=14-e;uint32_t r=m>>shift,rem=m&((1u<<shift)-1),mid=1u<<(shift-1);if(rem>mid||(rem==mid&&(r&1)))++r;return s|r;}
 uint32_t r=m>>13,rem=m&8191;if(rem>4096||(rem==4096&&(r&1))){if(++r==1024){r=0;if(++e==31)return s|0x7c00;}}
 return s|(e<<10)|r;
}
struct Q4 {uint16_t d,dmin;uint8_t scales[12],qs[128];};
struct Q6 {uint8_t ql[128],qh[64];int8_t scales[16];uint16_t d;};
static_assert(sizeof(Q4)==144&&sizeof(Q6)==210,"quant block layout");
static void scale_min(int j,const uint8_t*q,uint8_t&d,uint8_t&m){
 if(j<4){d=q[j]&63;m=q[j+4]&63;}else{d=(q[j+4]&15)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
void dequantize_scalar(const void*data,uint32_t type,float*out,int count){
 if(type==0){std::memcpy(out,data,size_t(count)*4);return;}
 if(count%256)throw std::runtime_error("unaligned quant row");
 if(type==12){
  auto*x=static_cast<const Q4*>(data);
  for(int i=0;i<count/256;i++){
   float d=half_to_float(x[i].d),mn=half_to_float(x[i].dmin);const uint8_t*q=x[i].qs;int is=0;
   for(int j=0;j<256;j+=64){uint8_t sc,m;scale_min(is,x[i].scales,sc,m);float d1=d*sc,m1=mn*m;scale_min(is+1,x[i].scales,sc,m);float d2=d*sc,m2=mn*m;
    for(int l=0;l<32;l++){*out++=d1*(q[l]&15)-m1;}
    for(int l=0;l<32;l++){*out++=d2*(q[l]>>4)-m2;}
    q+=32;is+=2;
   }
  }return;
 }
 if(type==14){
  auto*x=static_cast<const Q6*>(data);
  for(int i=0;i<count/256;i++){
   float d=half_to_float(x[i].d);const uint8_t*ql=x[i].ql,*qh=x[i].qh;const int8_t*sc=x[i].scales;
   for(int n=0;n<256;n+=128){for(int l=0;l<32;l++){int is=l/16;
    int a=int((ql[l]&15)|(((qh[l]>>0)&3)<<4))-32,b=int((ql[l+32]&15)|(((qh[l]>>2)&3)<<4))-32;
    int c=int((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32,e=int((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
    out[l]=d*sc[is]*a;out[l+32]=d*sc[is+2]*b;out[l+64]=d*sc[is+4]*c;out[l+96]=d*sc[is+6]*e;
   }out+=128;ql+=64;qh+=32;sc+=8;}
  }return;
 }
 throw std::runtime_error("unsupported quant type");
}
#ifdef __ARM_NEON
static void q4_store8(float*out,uint8x8_t q,float scale,float bias){
 uint16x8_t u=vmovl_u8(q);
 float32x4_t a=vcvtq_f32_u32(vmovl_u16(vget_low_u16(u))),b=vcvtq_f32_u32(vmovl_u16(vget_high_u16(u)));
 vst1q_f32(out,vsubq_f32(vmulq_n_f32(a,scale),vdupq_n_f32(bias)));
 vst1q_f32(out+4,vsubq_f32(vmulq_n_f32(b,scale),vdupq_n_f32(bias)));
}
static void q6_store8(float*out,uint8x8_t q,float scale){
 int16x8_t u=vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(q)),vdupq_n_s16(32));
 vst1q_f32(out,vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(u))),scale));
 vst1q_f32(out+4,vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(u))),scale));
}
#endif
void dequantize(const void*data,uint32_t type,float*out,int count){
 if(type==0){std::memcpy(out,data,size_t(count)*4);return;}
#ifdef __ARM_NEON
 if(count%256)throw std::runtime_error("unaligned quant row");
 if(type==12){
  auto*x=static_cast<const Q4*>(data);uint8x8_t mask=vdup_n_u8(15);
  for(int i=0;i<count/256;i++){
   float d=half_to_float(x[i].d),mn=half_to_float(x[i].dmin);
   for(int j=0;j<4;j++){
    uint8_t sc,m;scale_min(j*2,x[i].scales,sc,m);float d1=d*sc,m1=mn*m;
    scale_min(j*2+1,x[i].scales,sc,m);float d2=d*sc,m2=mn*m;
    for(int k=0;k<32;k+=8){uint8x8_t q=vld1_u8(x[i].qs+j*32+k);
     q4_store8(out+j*64+k,vand_u8(q,mask),d1,m1);
     q4_store8(out+j*64+32+k,vshr_n_u8(q,4),d2,m2);
    }
   }out+=256;
  }return;
 }
 if(type==14){
  auto*x=static_cast<const Q6*>(data);uint8x8_t lo=vdup_n_u8(15),hi=vdup_n_u8(48);
  for(int i=0;i<count/256;i++){
   float d=half_to_float(x[i].d);
   for(int n=0;n<2;n++)for(int k=0;k<32;k+=8){
    uint8x8_t a=vld1_u8(x[i].ql+n*64+k),b=vld1_u8(x[i].ql+n*64+k+32),h=vld1_u8(x[i].qh+n*32+k);
    const int8_t*sc=x[i].scales+n*8;int group=k/16;float*o=out+n*128+k;
    q6_store8(o,vorr_u8(vand_u8(a,lo),vand_u8(vshl_n_u8(h,4),hi)),d*sc[group]);
    q6_store8(o+32,vorr_u8(vand_u8(b,lo),vand_u8(vshl_n_u8(h,2),hi)),d*sc[group+2]);
    q6_store8(o+64,vorr_u8(vshr_n_u8(a,4),vand_u8(h,hi)),d*sc[group+4]);
    q6_store8(o+96,vorr_u8(vshr_n_u8(b,4),vand_u8(vshr_n_u8(h,2),hi)),d*sc[group+6]);
   }out+=256;
  }return;
 }
#endif
 dequantize_scalar(data,type,out,count);
}

// Preserve the original four-lane accumulation order while avoiding a float
// scratch row. Separate multiply/subtract keeps decoded weights unchanged.
float quantized_dot(const void*data,uint32_t type,const float*input,int count){
#ifdef __ARM_NEON
 if(count%256)throw std::runtime_error("unaligned quant dot");
 float32x4_t sum=vdupq_n_f32(0);
 auto accumulate=[&](uint8x8_t packed,float scale,float bias,bool centered,const float*in){
  uint16x8_t u=vmovl_u8(packed);float32x4_t a,b;
  if(centered){int16x8_t v=vsubq_s16(vreinterpretq_s16_u16(u),vdupq_n_s16(32));
   a=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))),scale);
   b=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v))),scale);
  }else{
   a=vsubq_f32(vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u))),scale),vdupq_n_f32(bias));
   b=vsubq_f32(vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u))),scale),vdupq_n_f32(bias));
  }
  sum=vmlaq_f32(sum,a,vld1q_f32(in));sum=vmlaq_f32(sum,b,vld1q_f32(in+4));
 };
 if(type==12){
  auto*x=static_cast<const Q4*>(data);
  for(int block=0;block<count/256;block++){
   float d=half_to_float(x[block].d),mn=half_to_float(x[block].dmin);
   for(int j=0;j<4;j++)for(int half=0;half<2;half++){
    uint8_t sc,m;scale_min(j*2+half,x[block].scales,sc,m);
    for(int k=0;k<32;k+=8){uint8x8_t q=vld1_u8(x[block].qs+j*32+k);
     q=half?vshr_n_u8(q,4):vand_u8(q,vdup_n_u8(15));
     accumulate(q,d*sc,mn*m,false,input+block*256+j*64+half*32+k);
    }
   }
  }
 }else if(type==14){
  auto*x=static_cast<const Q6*>(data);
  for(int block=0;block<count/256;block++){
   float d=half_to_float(x[block].d);
   for(int n=0;n<2;n++)for(int group=0;group<4;group++)for(int k=0;k<32;k+=8){
    uint8x8_t l=vld1_u8(x[block].ql+n*64+(group%2)*32+k),h=vld1_u8(x[block].qh+n*32+k);
    l=group>=2?vshr_n_u8(l,4):vand_u8(l,vdup_n_u8(15));
    h=vshl_u8(h,vdup_n_s8(-2*group));h=vshl_n_u8(vand_u8(h,vdup_n_u8(3)),4);
    accumulate(vorr_u8(l,h),d*x[block].scales[n*8+group*2+k/16],0,true,input+block*256+n*128+group*32+k);
   }
  }
 }else throw std::runtime_error("unsupported quant dot type");
 float32x2_t t=vadd_f32(vget_low_f32(sum),vget_high_f32(sum));t=vpadd_f32(t,t);return vget_lane_f32(t,0);
#else
 std::vector<float> row(count);dequantize(data,type,row.data(),count);
 float lane[4]={0,0,0,0};for(int i=0;i<count;i++)lane[i%4]+=row[i]*input[i];
 return (lane[0]+lane[2])+(lane[1]+lane[3]);
#endif
}


// Each token retains the single-token four-lane accumulation order. Decode once
// per register group; templates let the compiler specialize the accumulator count.
#ifdef __ARM_NEON
template<int N> static void fused_batch(const void*data,uint32_t type,const float* const*input,float*out,int count){
 if(count%256)throw std::runtime_error("unaligned quant dot");
 float32x4_t sums[N];for(int t=0;t<N;t++)sums[t]=vdupq_n_f32(0);
 auto accumulate=[&](uint8x8_t packed,float scale,float bias,bool centered,int offset){
  uint16x8_t u=vmovl_u8(packed);float32x4_t a,b;
  if(centered){int16x8_t v=vsubq_s16(vreinterpretq_s16_u16(u),vdupq_n_s16(32));
   a=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))),scale);
   b=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v))),scale);
  }else{
   a=vsubq_f32(vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u))),scale),vdupq_n_f32(bias));
   b=vsubq_f32(vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u))),scale),vdupq_n_f32(bias));
  }
  for(int t=0;t<N;t++){const float*in=input[t]+offset;sums[t]=vmlaq_f32(sums[t],a,vld1q_f32(in));sums[t]=vmlaq_f32(sums[t],b,vld1q_f32(in+4));}
 };
 if(type==12){
  auto*x=static_cast<const Q4*>(data);
  for(int block=0;block<count/256;block++){
   float d=half_to_float(x[block].d),mn=half_to_float(x[block].dmin);
   for(int j=0;j<4;j++)for(int half=0;half<2;half++){
    uint8_t sc,m;scale_min(j*2+half,x[block].scales,sc,m);
    for(int k=0;k<32;k+=8){uint8x8_t q=vld1_u8(x[block].qs+j*32+k);
     q=half?vshr_n_u8(q,4):vand_u8(q,vdup_n_u8(15));
     accumulate(q,d*sc,mn*m,false,block*256+j*64+half*32+k);
    }
   }
  }
 }else if(type==14){
  auto*x=static_cast<const Q6*>(data);
  for(int block=0;block<count/256;block++){
   float d=half_to_float(x[block].d);
   for(int n=0;n<2;n++)for(int group=0;group<4;group++)for(int k=0;k<32;k+=8){
    uint8x8_t l=vld1_u8(x[block].ql+n*64+(group%2)*32+k),h=vld1_u8(x[block].qh+n*32+k);
    l=group>=2?vshr_n_u8(l,4):vand_u8(l,vdup_n_u8(15));
    h=vshl_u8(h,vdup_n_s8(-2*group));h=vshl_n_u8(vand_u8(h,vdup_n_u8(3)),4);
    accumulate(vorr_u8(l,h),d*x[block].scales[n*8+group*2+k/16],0,true,block*256+n*128+group*32+k);
   }
  }
 }else throw std::runtime_error("unsupported quant dot type");
 for(int i=0;i<N;i++){float32x2_t t=vadd_f32(vget_low_f32(sums[i]),vget_high_f32(sums[i]));t=vpadd_f32(t,t);out[i]=vget_lane_f32(t,0);}
}
#endif
void quantized_dot_batch(const void*data,uint32_t type,const float* const*input,float*out,int count,int batch){
 if(batch<1||batch>8)throw std::runtime_error("invalid fused batch width");
#ifdef __ARM_NEON
 switch(batch){
 case 1:fused_batch<1>(data,type,input,out,count);return;
 case 2:fused_batch<2>(data,type,input,out,count);return;
 case 3:fused_batch<3>(data,type,input,out,count);return;
 case 4:fused_batch<4>(data,type,input,out,count);return;
 case 5:fused_batch<5>(data,type,input,out,count);return;
 case 6:fused_batch<6>(data,type,input,out,count);return;
 case 7:fused_batch<7>(data,type,input,out,count);return;
 case 8:fused_batch<8>(data,type,input,out,count);return;

 }
#else
 for(int i=0;i<batch;i++)out[i]=quantized_dot(data,type,input[i],count);
#endif
}

// Quantize each input group once, outside the matrix row loop.
void quantize_activation(const float* input,ActivationBlock* blocks,int count,int batch){
 for(int offset=0;offset<count;offset+=32){
  auto& block=blocks[(offset/32)*batch];
  float peak=0;
  for(int i=0;i<32;i++)peak=std::max(peak,std::fabs(input[offset+i]));
  block.scale=peak/127.f;
  const float inverse=block.scale>0?1.f/block.scale:0.f;
  block.sum=0;
  for(int i=0;i<32;i++){
   const int value=int(std::round(input[offset+i]*inverse));
   block.values[i]=int8_t(value);
   block.sum+=value;
  }
 }
}
template<int Fixed> static void q4_activation_dot_tile(const void* weights,const ActivationBlock* input,float* out,int count,int width){
 const int batch=Fixed?Fixed:width;
 const auto* row=static_cast<const Q4*>(weights);
 float totals[16]={};
 for(int b=0;b<count/256;b++){
  const float scale=half_to_float(row[b].d),bias=half_to_float(row[b].dmin);
  for(int group=0;group<8;group++){
   uint8_t sc,m;
   scale_min(group,row[b].scales,sc,m);
   const auto* packed=row[b].qs+(group/2)*32;
   const auto* activation=input+(b*8+group)*batch;
   const float d=scale*sc,min=bias*m;
#ifdef __ARM_NEON
   uint8x16_t lo=vld1q_u8(packed),hi=vld1q_u8(packed+16);
   if(group%2){lo=vshrq_n_u8(lo,4);hi=vshrq_n_u8(hi,4);}
   else{lo=vandq_u8(lo,vdupq_n_u8(15));hi=vandq_u8(hi,vdupq_n_u8(15));}
   const int8x16_t q0=vreinterpretq_s8_u8(lo),q1=vreinterpretq_s8_u8(hi);
   #pragma GCC unroll 8
   for(int t=0;t<batch;t++){
    const auto& a=activation[t];
    const int8x16_t a0=vld1q_s8(a.values),a1=vld1q_s8(a.values+16);
    int16x8_t products=vmull_s8(vget_low_s8(q0),vget_low_s8(a0));
    products=vmlal_s8(products,vget_high_s8(q0),vget_high_s8(a0));
    products=vmlal_s8(products,vget_low_s8(q1),vget_low_s8(a1));
    products=vmlal_s8(products,vget_high_s8(q1),vget_high_s8(a1));
    const int32x4_t sums=vpaddlq_s16(products);
    int32x2_t pair=vadd_s32(vget_low_s32(sums),vget_high_s32(sums));
    pair=vpadd_s32(pair,pair);
    totals[t]+=a.scale*(d*vget_lane_s32(pair,0)-min*a.sum);
   }
#else
   #pragma GCC unroll 8
   for(int t=0;t<batch;t++){
    const auto& a=activation[t];
    int dot=0;
    for(int i=0;i<32;i++)dot+=((packed[i]>>((group%2)*4))&15)*a.values[i];
    totals[t]+=a.scale*(d*dot-min*a.sum);
   }
#endif
  }
 }
 std::copy_n(totals,batch,out);
}
void q4_activation_dot(const void* weights,const ActivationBlock* input,float* out,int count,int batch){
 if(batch==8)q4_activation_dot_tile<8>(weights,input,out,count,batch);
 else q4_activation_dot_tile<0>(weights,input,out,count,batch);
}

template<int Fixed> static void q6_activation_dot_tile(const void* weights,const ActivationBlock* input,float* out,int count,int width){
 const int batch=Fixed?Fixed:width;
 const auto* row=static_cast<const Q6*>(weights);
 float totals[16]={};
 for(int b=0;b<count/256;b++){
  const float scale=half_to_float(row[b].d);
  for(int half=0;half<2;half++)for(int group=0;group<4;group++){
   const auto* low=row[b].ql+half*64+(group%2)*32;
   const auto* high=row[b].qh+half*32;
   const auto* activation=input+(b*8+half*4+group)*batch;
   const float d0=scale*row[b].scales[half*8+group*2];
   const float d1=scale*row[b].scales[half*8+group*2+1];
#ifdef __ARM_NEON
   const auto unpack=[&](int offset){
    uint8x16_t l=vld1q_u8(low+offset),h=vld1q_u8(high+offset);
    l=group>=2?vshrq_n_u8(l,4):vandq_u8(l,vdupq_n_u8(15));
    h=vshlq_u8(h,vdupq_n_s8(-2*group));
    h=vshlq_n_u8(vandq_u8(h,vdupq_n_u8(3)),4);
    return vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l,h)),vdupq_n_s8(32));
   };
   const auto q0=unpack(0),q1=unpack(16);
   const auto dot16=[](int8x16_t q,int8x16_t a){
    auto product=vmull_s8(vget_low_s8(q),vget_low_s8(a));
    product=vmlal_s8(product,vget_high_s8(q),vget_high_s8(a));
    const auto sums=vpaddlq_s16(product);
    auto pair=vadd_s32(vget_low_s32(sums),vget_high_s32(sums));
    pair=vpadd_s32(pair,pair);
    return vget_lane_s32(pair,0);
   };
   #pragma GCC unroll 8
   for(int t=0;t<batch;t++){
    const auto& a=activation[t];
    totals[t]+=a.scale*(d0*dot16(q0,vld1q_s8(a.values))+d1*dot16(q1,vld1q_s8(a.values+16)));
   }
#else
   #pragma GCC unroll 8
   for(int t=0;t<batch;t++){
    const auto& a=activation[t];
    int dot[2]={0,0};
    for(int k=0;k<32;k++){
     const int q=((low[k]>>(group>=2?4:0))&15)|(((high[k]>>(2*group))&3)<<4);
     dot[k/16]+=(q-32)*a.values[k];
    }
    totals[t]+=a.scale*(d0*dot[0]+d1*dot[1]);
   }
#endif
  }
 }
 std::copy_n(totals,batch,out);
}
void q6_activation_dot(const void* weights,const ActivationBlock* input,float* out,int count,int batch){
 if(batch==8)q6_activation_dot_tile<8>(weights,input,out,count,batch);
 else q6_activation_dot_tile<0>(weights,input,out,count,batch);
}

#ifdef __ARM_NEON
// Each SIMD lane owns one token. Two partial sums keep products within int16.
static int16x8_t q4_dot_eight(const int8_t (*lanes)[8],int8x8_t weight){
 auto product=vmull_s8(vld1_s8(lanes[0]),vdup_lane_s8(weight,0));
 product=vmlal_s8(product,vld1_s8(lanes[1]),vdup_lane_s8(weight,1));
 product=vmlal_s8(product,vld1_s8(lanes[2]),vdup_lane_s8(weight,2));
 product=vmlal_s8(product,vld1_s8(lanes[3]),vdup_lane_s8(weight,3));
 product=vmlal_s8(product,vld1_s8(lanes[4]),vdup_lane_s8(weight,4));
 product=vmlal_s8(product,vld1_s8(lanes[5]),vdup_lane_s8(weight,5));
 product=vmlal_s8(product,vld1_s8(lanes[6]),vdup_lane_s8(weight,6));
 return vmlal_s8(product,vld1_s8(lanes[7]),vdup_lane_s8(weight,7));
}
void q4_dot_lane8(const void* weights,const Q4InputTile* input,float* out,int count){
 const auto* row=static_cast<const Q4*>(weights);
 float32x4_t total0=vdupq_n_f32(0),total1=vdupq_n_f32(0);
 for(int block=0;block<count/256;block++){
  const float scale=half_to_float(row[block].d),offset=half_to_float(row[block].dmin);
  for(int group=0;group<8;group++){
   uint8_t sc,m;
   scale_min(group,row[block].scales,sc,m);
   const auto& tile=input[block*8+group];
   const auto* packed=row[block].qs+(group/2)*32;
   const auto unpack=[&](int k){
    const auto raw=vld1_u8(packed+k);
    return vreinterpret_s8_u8(group%2?vshr_n_u8(raw,4):vand_u8(raw,vdup_n_u8(15)));
   };
   const auto dot0=vaddq_s16(q4_dot_eight(tile.lanes,unpack(0)),
                            q4_dot_eight(tile.lanes+8,unpack(8)));
   const auto dot1=vaddq_s16(q4_dot_eight(tile.lanes+16,unpack(16)),
                            q4_dot_eight(tile.lanes+24,unpack(24)));
   const auto sum0=vaddl_s16(vget_low_s16(dot0),vget_low_s16(dot1));
   const auto sum1=vaddl_s16(vget_high_s16(dot0),vget_high_s16(dot1));
   const float d=scale*sc,bias=offset*m;
   const auto a0=vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(sum0),d),vmulq_n_f32(vld1q_f32(tile.sum),bias));
   const auto a1=vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(sum1),d),vmulq_n_f32(vld1q_f32(tile.sum+4),bias));
   total0=vaddq_f32(total0,vmulq_f32(a0,vld1q_f32(tile.scale)));
   total1=vaddq_f32(total1,vmulq_f32(a1,vld1q_f32(tile.scale+4)));
  }
 }
 vst1q_f32(out,total0);
 vst1q_f32(out+4,total1);
}
void q6_dot_lane8(const void* weights,const Q4InputTile* input,float* out,int count){
 const auto* row=static_cast<const Q6*>(weights);
 float32x4_t lo=vdupq_n_f32(0),hi=vdupq_n_f32(0);
 for(int block=0;block<count/256;block++){
  const float scale=half_to_float(row[block].d);
  for(int half=0;half<2;half++)for(int group=0;group<4;group++){
   const auto* low=row[block].ql+half*64+(group%2)*32;
   const auto* high=row[block].qh+half*32;
   const auto& tile=input[block*8+half*4+group];
   const auto unpack=[&](int k){
    auto l=vld1_u8(low+k);
    l=group>=2?vshr_n_u8(l,4):vand_u8(l,vdup_n_u8(15));
    auto h=vshl_u8(vld1_u8(high+k),vdup_n_s8(-2*group));
    h=vshl_n_u8(vand_u8(h,vdup_n_u8(3)),4);
    return vsub_s8(vreinterpret_s8_u8(vorr_u8(l,h)),vdup_n_s8(32));
   };
   // Widen each eight-product partial before adding to avoid signed overflow.
   const auto a=q4_dot_eight(tile.lanes,unpack(0));
   const auto b=q4_dot_eight(tile.lanes+8,unpack(8));
   const auto c=q4_dot_eight(tile.lanes+16,unpack(16));
   const auto d=q4_dot_eight(tile.lanes+24,unpack(24));
   const float d0=scale*row[block].scales[half*8+group*2];
   const float d1=scale*row[block].scales[half*8+group*2+1];
   const auto l=vaddq_f32(vmulq_n_f32(vcvtq_f32_s32(vaddl_s16(vget_low_s16(a),vget_low_s16(b))),d0),
                          vmulq_n_f32(vcvtq_f32_s32(vaddl_s16(vget_low_s16(c),vget_low_s16(d))),d1));
   const auto h=vaddq_f32(vmulq_n_f32(vcvtq_f32_s32(vaddl_s16(vget_high_s16(a),vget_high_s16(b))),d0),
                          vmulq_n_f32(vcvtq_f32_s32(vaddl_s16(vget_high_s16(c),vget_high_s16(d))),d1));
   lo=vaddq_f32(lo,vmulq_f32(l,vld1q_f32(tile.scale)));
   hi=vaddq_f32(hi,vmulq_f32(h,vld1q_f32(tile.scale+4)));
  }
 }
 vst1q_f32(out,lo);
 vst1q_f32(out+4,hi);
}
#endif

}
