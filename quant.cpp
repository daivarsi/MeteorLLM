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

}
