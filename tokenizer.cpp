#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
namespace vitaqwen {
#include "unicode_ranges.inc"
static uint32_t read32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
static void check(bool b,const char*m){if(!b)throw std::runtime_error(m);}
template<size_t N>static bool in_range(uint32_t cp,const uint32_t(&ranges)[N][2]){size_t a=0,b=N;while(a<b){size_t m=(a+b)/2;if(cp<ranges[m][0])b=m;else if(cp>ranges[m][1])a=m+1;else return true;}return false;}
static bool letter(uint32_t cp){return in_range(cp,letters);}
static bool number(uint32_t cp){return in_range(cp,numbers);}
static bool space(uint32_t c){return (c>=9&&c<=13)||c==32||c==133||c==160||c==5760||(c>=8192&&c<=8202)||c==8232||c==8233||c==8239||c==8287||c==12288;}
static uint32_t cp_at(const std::string&s,size_t p,size_t&n){
 n=1;if(p>=s.size())return 0;uint8_t c=s[p];if(c<128)return c;
 int len=c>=0xf0?4:c>=0xe0?3:c>=0xc2?2:0;if(!len||p+len>s.size())return 0xfffd;
 uint32_t value=c&((1u<<(7-len))-1);for(int i=1;i<len;i++){uint8_t b=s[p+i];if((b&192)!=128)return 0xfffd;value=(value<<6)|(b&63);}
 if(value>0x10ffff||(value>=0xd800&&value<=0xdfff)||(len==2&&value<128)||(len==3&&value<2048)||(len==4&&value<65536))return 0xfffd;
 n=len;return value;
}
static size_t split(const std::string&s,size_t p){
 size_t n;uint32_t cp=cp_at(s,p,n);if(cp=='\''){
  for(const char*tail:{"s","t","re","ve","m","ll","d"}){size_t len=std::strlen(tail);bool yes=p+1+len<=s.size();for(size_t j=0;yes&&j<len;j++){char c=s[p+1+j];if(c>='A'&&c<='Z')c+=32;if(c!=tail[j])yes=false;}if(yes)return p+1+len;}
 }
 size_t q=p;if(!letter(cp)&&!number(cp)&&cp!='\r'&&cp!='\n')q+=n;
 size_t qn;uint32_t qc=cp_at(s,q,qn);
 if(q<s.size()&&letter(qc)){do{q+=qn;qc=cp_at(s,q,qn);}while(q<s.size()&&letter(qc));return q;}
 if(number(cp))return p+n;
 q=p;if(cp==' '&&p+1<s.size())++q;qc=cp_at(s,q,qn);
 if(q<s.size()&&!space(qc)&&!letter(qc)&&!number(qc)){do{q+=qn;qc=cp_at(s,q,qn);}while(q<s.size()&&!space(qc)&&!letter(qc)&&!number(qc));while(q<s.size()&&(s[q]=='\r'||s[q]=='\n'))++q;return q;}
 q=p;size_t last_newline=p,last_cp=p;
 while(q<s.size()){qc=cp_at(s,q,qn);if(!space(qc))break;last_cp=q;q+=qn;if(qc=='\r'||qc=='\n')last_newline=q;}
 if(last_newline>p)return last_newline;
 if(q>p)return q<s.size()&&last_cp>p?last_cp:q;
 return p+n;
}
void Tokenizer::load(const std::string&path){
 FILE*f=std::fopen(path.c_str(),"rb");check(f,"tokenizer open failed");std::fseek(f,0,SEEK_END);long size=std::ftell(f);std::rewind(f);
 if(size<64||size>16*1024*1024){std::fclose(f);throw std::runtime_error("tokenizer size bounds");}data.resize(size);size_t got=std::fread(data.data(),1,size,f);std::fclose(f);check(got==size_t(size)&&!std::memcmp(data.data(),"MTOK0001",8),"tokenizer header/read");
 uint32_t*fields[]={&count,&merge_count,&special_count,&offset_base,&blob_base,&blob_size,&byte_base,&special_base,&merge_base};for(int i=0;i<9;i++)*fields[i]=read32(data.data()+8+4*i);
 check(count>256&&count<=300000&&merge_count<=300000&&special_count<=1024,"tokenizer counts");
 check(read32(data.data()+44)==uint32_t(size)&&read32(data.data()+48)==0&&offset_base==64&&blob_base==offset_base+4*(count+1),"tokenizer offsets");
 check(uint64_t(blob_base)+blob_size<=uint64_t(size)&&byte_base>=blob_base+blob_size&&byte_base-(blob_base+blob_size)<4&&special_base==byte_base+1024&&uint64_t(special_base)+special_count*4<=uint64_t(size),"tokenizer blob bounds");
 check(merge_base>=special_base+special_count*4&&merge_base-(special_base+special_count*4)<8&&uint64_t(merge_base)+uint64_t(merge_count)*16==uint64_t(size),"merge table bounds");
 uint32_t prev=0;for(uint32_t i=0;i<=count;i++){uint32_t at=read32(data.data()+offset_base+4*i);check(at>=prev&&at<=blob_size,"token offset bounds");prev=at;}check(prev==blob_size,"token blob tail");
 for(uint32_t i=0;i<256;i++)check(read32(data.data()+byte_base+4*i)<count,"byte token bounds");
 for(uint32_t i=0;i<special_count;i++)check(read32(data.data()+special_base+4*i)<count,"special token bounds");
 uint64_t previous=0;for(uint32_t i=0;i<merge_count;i++){const uint8_t*p=data.data()+merge_base+16*i;uint32_t a=read32(p),b=read32(p+4),to=read32(p+8),rank=read32(p+12);uint64_t key=(uint64_t(a)<<32)|b;check(a<count&&b<count&&to<count&&rank<merge_count&&(!i||key>previous),"invalid merge entry");previous=key;}
}
std::string Tokenizer::decode(int token)const{check(token>=0&&uint32_t(token)<count,"decode token bounds");uint32_t a=read32(data.data()+offset_base+4*token),b=read32(data.data()+offset_base+4*(token+1));return std::string(reinterpret_cast<const char*>(data.data()+blob_base+a),b-a);}
bool Tokenizer::is_special(int token)const{
 for(uint32_t i=0;i<special_count;i++)
  if(int(read32(data.data()+special_base+4*i))==token)return true;
 return false;
}
int Tokenizer::special_at(const std::string&s,size_t p,size_t&len)const{int found=-1;len=0;for(uint32_t i=0;i<special_count;i++){int id=read32(data.data()+special_base+4*i);std::string text=decode(id);if(text.size()>len&&s.compare(p,text.size(),text)==0){found=id;len=text.size();}}return found;}
void Tokenizer::bpe(const std::string&s,size_t start,size_t end,std::vector<int>&out)const{
 std::vector<uint32_t>ids;for(size_t p=start;p<end;p++)ids.push_back(read32(data.data()+byte_base+4*uint8_t(s[p])));
 while(ids.size()>1){uint32_t best=UINT32_MAX,to=0;size_t where=0;
  for(size_t i=0;i+1<ids.size();i++){uint64_t key=(uint64_t(ids[i])<<32)|ids[i+1];size_t a=0,b=merge_count;while(a<b){size_t m=(a+b)/2;const uint8_t*p=data.data()+merge_base+16*m;uint64_t mk=(uint64_t(read32(p))<<32)|read32(p+4);if(mk<key)a=m+1;else b=m;}if(a<merge_count){const uint8_t*p=data.data()+merge_base+16*a;if(read32(p)==ids[i]&&read32(p+4)==ids[i+1]&&read32(p+12)<best){best=read32(p+12);to=read32(p+8);where=i;}}}
  if(best==UINT32_MAX)break;
  ids[where]=to;ids.erase(ids.begin()+where+1);
 }
 for(uint32_t id:ids)out.push_back(id);
}
std::vector<int> Tokenizer::encode(const std::string&text)const{
 check(!data.empty()&&text.size()<=8192,"tokenizer not loaded or input too long");std::vector<int>out;size_t p=0;
 // Control tokens divide ordinary text before regex splitting. In particular,
 // whitespace before a control token is trailing whitespace in that segment.
 while(p<text.size()){
  size_t len;int id=special_at(text,p,len);if(id>=0){out.push_back(id);p+=len;continue;}
  size_t boundary=text.size();for(size_t k=p+1;k<text.size();k++){size_t n;if(special_at(text,k,n)>=0){boundary=k;break;}}
  std::string segment=text.substr(p,boundary-p);size_t at=0;
  while(at<segment.size()){size_t end=split(segment,at);bpe(segment,at,end,out);at=end;}
  p=boundary;
 }
 return out;
}
}
