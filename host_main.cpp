#include "engine.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
int main(int argc,char**argv){
 try{
  if(argc==2&&std::string(argv[1])=="--half-test"){
   for(unsigned h=0;h<65536;h++){float f=vitaqwen::half_to_float(h);unsigned back=vitaqwen::float_to_half(f);if(!std::isnan(f)&&back!=h)throw std::runtime_error("half round-trip failed");}
   std::cout<<"65536 half patterns checked\n";return 0;
  }
  if(argc==3&&std::string(argv[2])=="--kernel-check"){vitaqwen::Engine e;e.load(argv[1]);std::cout<<"Kernel parity values="<<e.validate_quant_kernels()<<"\n";return 0;}
  if(argc==3&&std::string(argv[2])=="--load-only"){vitaqwen::Engine e;e.load(argv[1]);std::cout<<"Model validated\n";return 0;}
  if(argc==3&&std::string(argv[2])=="--matrix-test"){vitaqwen::Engine e;e.load(argv[1]);e.benchmark_matrices([](const std::string&s){std::cout<<s<<"\n";});return 0;}
  if(argc==3&&std::string(argv[2])=="--batch-test"){vitaqwen::Engine e;e.load(argv[1]);e.validate_batch([](const std::string&s){std::cout<<s<<"\n";});return 0;}
  if(argc==3&&std::string(argv[2])=="--extended-test"){vitaqwen::Engine e;e.load(argv[1]);auto report=[](const std::string&s){std::cout<<s<<std::endl;};e.benchmark_extended(report);e.validate_extended(report);e.benchmark_output(report);return 0;}
  if(argc==3&&std::string(argv[2])=="--attention-test"){vitaqwen::Engine e;e.load(argv[1]);e.validate_extended([](const std::string&s){std::cout<<s<<std::endl;});return 0;}
  if(argc<4){std::cerr<<"model-directory token-ids-file logits-file [generated-token-count]\n";return 2;}
  vitaqwen::Engine engine;auto start=std::chrono::steady_clock::now();engine.load(argv[1]);if(argc>5)engine.set_workers(std::stoi(argv[5]));
  if(argc>7)engine.set_batch_kernel(std::stoi(argv[7]));if(argc>8)engine.set_output_rows(std::stoi(argv[8]));
  if(argc>9)engine.set_parallel_attention(std::stoi(argv[9])!=0);
  std::ifstream file(argv[2]);if(!file)throw std::runtime_error("tokens open failed");std::vector<int>tokens;int id;while(file>>id)tokens.push_back(id);if(tokens.empty())throw std::runtime_error("empty prompt");
  const std::vector<float>*scores=nullptr;int pos=0;
  int batch=argc>6?std::stoi(argv[6]):1;if(batch<1||batch>16)throw std::runtime_error("invalid batch size");while(pos<int(tokens.size())){int n=std::min(batch,int(tokens.size())-pos);scores=&engine.forward_batch(std::vector<int>(tokens.begin()+pos,tokens.begin()+pos+n),pos,pos+n==int(tokens.size()));pos+=n;}
  std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(scores->data()),scores->size()*4);if(!out)throw std::runtime_error("logits write failed");
  std::vector<int>order(scores->size());std::iota(order.begin(),order.end(),0);std::partial_sort(order.begin(),order.begin()+10,order.end(),[&](int a,int b){return (*scores)[a]>(*scores)[b];});
  std::cout<<"TOP";for(int i=0;i<10;i++)std::cout<<" "<<order[i]<<":"<<(*scores)[order[i]];std::cout<<"\nGENERATED";
  int limit=argc>4?std::stoi(argv[4]):1;
  for(int n=0;n<limit&&pos<engine.config().context;n++){int next=std::max_element(scores->begin(),scores->end())-scores->begin();std::cout<<" "<<next;std::cout.flush();if(next==engine.config().eos)break;if(n+1<limit)scores=&engine.forward(next,pos++);}
  std::cout<<"\nresident_bytes="<<engine.resident_bytes()<<" embedding_read_bytes="<<engine.embedding_read_bytes<<" "+engine.stage_metrics()+" total_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
 }catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
}
