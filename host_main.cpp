#include "engine.h"
#include "tokenizer.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
static std::string quoted(const std::string&s){std::string out="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c<32){char b[7];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b;}else out+=c;}return out+'"';}
int main(int argc,char**argv){
 if(argc==2&&std::string(argv[1])=="--half-test"){for(unsigned h=0;h<65536;h++){float f=vitaqwen::half_to_float(h);if(!std::isnan(f)&&vitaqwen::float_to_half(f)!=h)return 1;}return 0;}
 if(argc!=4)return 2;
 try{
  vitaqwen::Engine engine;engine.load(argv[1]);if(std::getenv("OD_REFERENCE")){engine.int8=false;engine.q4_head=false;engine.fast_output=false;}else if(std::getenv("OD_FULL"))engine.fast_output=false;if(std::getenv("OD_NOINT8"))engine.int8=false;if(std::getenv("OD_Q6"))engine.q4_head=false;engine.set_workers(3);engine.set_parallel_attention(true);engine.set_batch_kernel(2);engine.set_output_rows(256);
  vitaqwen::Tokenizer tokenizer;tokenizer.load(std::string(argv[1])+"/tokenizer.bin");
  const std::string system="<|im_start|>system\nYou are a helpful assistant. Follow the instructions and answer concisely. If you do not know, say so.<|im_end|>\n";
  const auto prefix=tokenizer.encode(system+"<|im_start|>user\n");engine.forward_batch(prefix,0,false);engine.save_prefix();
  std::ifstream cases(argv[2]);std::ofstream out(argv[3]);if(!cases||!out)throw std::runtime_error("case/output file open failed");
  std::string line;
  while(std::getline(cases,line)){
   if(!line.empty()&&line.back()=='\r')line.pop_back();const auto a=line.find('\t'),b=line.find('\t',a+1);if(a==line.npos||b==line.npos)throw std::runtime_error("bad case line");
   const auto id=line.substr(0,a),prompt=line.substr(b+1);const int limit=std::stoi(line.substr(a+1,b-a-1));
   const auto ids=tokenizer.encode(system+"<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
   if(!std::equal(prefix.begin(),prefix.end(),ids.begin())||ids.size()-prefix.size()>128)throw std::runtime_error("unexpected prefix/batch");
   engine.restore_prefix();engine.reset_metrics();
   const auto*scores=&engine.forward_batch(std::vector<int>(ids.begin()+prefix.size(),ids.end()),int(prefix.size()),true);
   std::string answer,finish="length";int pos=int(ids.size()),generated=0;
   for(int n=0;n<limit;n++){
    const int token=int(std::max_element(scores->begin(),scores->end())-scores->begin());++generated;
    if(token==engine.config().eos||token==248044){finish="stop";break;}
    answer+=tokenizer.decode(token);if(n+1<limit)scores=&engine.forward(token,pos++,true);
   }
   out<<"{\"case\":"<<quoted(id)<<",\"prompt\":"<<quoted(prompt)<<",\"text\":"<<quoted(answer)<<",\"finish_reason\":"<<quoted(finish)<<",\"generated\":"<<generated<<",\"cached_tokens\":"<<prefix.size()<<",\"output_full_scans\":"<<engine.output_full_scans<<"}\n";out.flush();
   std::printf("CUSTOM %s %s %s\n",id.c_str(),finish.c_str(),answer.c_str());std::fflush(stdout);
  }
 }catch(const std::exception&e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
 return 0;
}
