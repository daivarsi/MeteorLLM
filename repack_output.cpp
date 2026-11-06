#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc,char** argv){
 if(argc!=3)return 2;
 FILE* input=std::fopen(argv[1],"rb");
 FILE* output=std::fopen(argv[2],"wb");
 if(!input||!output)return 2;
 unsigned char source[210];
 size_t count=0;
 while(std::fread(source,sizeof(source),1,input)==1){
  float values[256];
  vitaqwen::dequantize_scalar(source,14,values,256);
  float steps[8],offsets[8];
  for(int g=0;g<8;g++){
   const auto ends=std::minmax_element(values+g*32,values+(g+1)*32);
   const float lower=std::min(0.f,*ends.first);
   offsets[g]=-lower;
   steps[g]=(*ends.second-lower)/15.f;
  }
  const uint16_t step16=vitaqwen::float_to_half(*std::max_element(steps,steps+8)/63.f);
  const uint16_t offset16=vitaqwen::float_to_half(*std::max_element(offsets,offsets+8)/63.f);
  const float step=vitaqwen::half_to_float(step16),offset=vitaqwen::half_to_float(offset16);
  unsigned char dest[144]={};
  dest[0]=step16&255;dest[1]=step16>>8;
  dest[2]=offset16&255;dest[3]=offset16>>8;
  auto* scales=dest+4;
  for(int g=0;g<8;g++){
   int d=step>0?std::clamp(int(std::round(steps[g]/step)),0,63):0;
   int m=offset>0?std::clamp(int(std::round(offsets[g]/offset)),0,63):0;
   if(g<4){scales[g]=d;scales[g+4]=m;}
   else{
    scales[g+4]=(d&15)|((m&15)<<4);
    scales[g-4]|=(d>>4)<<6;
    scales[g]|=(m>>4)<<6;
   }
   for(int k=0;k<32;k++){
    int q=d?std::clamp(int(std::round((values[g*32+k]+offset*m)/(step*d))),0,15):0;
    dest[16+(g/2)*32+k]|=q<<((g%2)*4);
   }
  }
  if(std::fwrite(dest,sizeof(dest),1,output)!=1)return 3;
  count++;
 }
 return std::fclose(output)!=0||count!=248320*4||std::ferror(input);
}
