#include "engine.h"
#include "tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
int main(int argc,char**argv){try{if(argc!=2)return 2;vitaqwen::Engine e;e.load(argv[1]);e.fast_output=false;vitaqwen::Tokenizer t;t.load(std::string(argv[1])+"/tokenizer.bin");auto ids=t.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");e.set_workers(3);auto reference=e.forward_batch(ids,0,true);auto continuation=e.forward(198,ids.size(),true);float maxerr=0;
 // Re-run the single-token path once, then compare the same continuation.
 e.reset();e.set_workers(1);const std::vector<float>*last=nullptr;for(size_t i=0;i<ids.size();i++)last=&e.forward(ids[i],i,i+1==ids.size());for(size_t i=0;i<reference.size();i++)maxerr=std::max(maxerr,std::abs((*last)[i]-reference[i]));if(maxerr>0.002f)throw std::runtime_error("batch prefill mismatch");std::printf("BATCH workers1/3 prefill max_abs=%g\n",maxerr);auto single=e.forward(198,ids.size(),true);maxerr=0;for(size_t i=0;i<single.size();i++)maxerr=std::max(maxerr,std::abs(single[i]-continuation[i]));if(maxerr>0.002f)throw std::runtime_error("batch/decode mismatch");std::printf("BATCH workers1/3 continuation max_abs=%g\n",maxerr);
 e.reset();e.set_workers(3);e.forward_batch(ids,0,false);e.save_prefix();auto a=e.forward(198,ids.size(),true);e.restore_prefix();auto b=e.forward(198,ids.size(),true);if(a!=b)throw std::runtime_error("disk prefix restore mismatch");std::puts("PREFIX restore full logits bit_exact=1");e.cache_output=false;e.restore_prefix();auto uncached=e.forward(198,ids.size(),true);if(a!=uncached)throw std::runtime_error("output cache mismatch");std::puts("OUTPUT_CACHE full logits bit_exact=1");
 e.cache_output=true;for(int workers:{4,3,1}){e.set_workers(workers);e.restore_prefix();if(e.forward(198,ids.size(),true)!=a)throw std::runtime_error("worker switch changed logits");}std::puts("WORKERS 3/4/3/1 full logits bit_exact=1");
 e.reset();e.cancelled=[](){return true;};bool stopped=false;try{e.forward(198,0,false);}catch(const std::runtime_error&x){stopped=std::string(x.what())=="Stopped";}if(!stopped)throw std::runtime_error("cancel failed");e.cancelled={};e.reset();std::puts("CANCEL reset recovery passed");
 }catch(const std::exception&e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}return 0;}
