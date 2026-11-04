#include "engine.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{if(argc!=3)return 2;std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);uint32_t type;
 while(in.read(reinterpret_cast<char*>(&type),4)){uint32_t n;in.read(reinterpret_cast<char*>(&n),4);if(!in||!n||n>65536||n%256||(type!=12&&type!=14))throw std::runtime_error("bad block test input");size_t bytes=n/256*(type==12?144:210);std::vector<uint8_t>b(bytes);std::vector<float>f(n);in.read(reinterpret_cast<char*>(b.data()),bytes);if(!in)throw std::runtime_error("short test input");vitaqwen::dequantize(b.data(),type,f.data(),n);out.write(reinterpret_cast<char*>(f.data()),n*4);}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
