#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <iterator>
int main(int argc,char**argv){try{if(argc<3)return 2;vitaqwen::Tokenizer t;t.load(argv[1]);std::ifstream f(argv[2],std::ios::binary);if(!f)throw std::runtime_error("input open");
 auto emit=[&](const std::string&s){auto ids=t.encode(s);for(int id:ids)std::cout<<id<<' ';std::cout<<'\n';};
 if(argc==4&&std::string(argv[3])=="--batch"){uint32_t n;while(f.read(reinterpret_cast<char*>(&n),4)){if(n>8192)throw std::runtime_error("input size");std::string s(n,'\0');if(!f.read(s.data(),n))throw std::runtime_error("short input");emit(s);}}
 else{std::string s((std::istreambuf_iterator<char>(f)),{});emit(s);}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
