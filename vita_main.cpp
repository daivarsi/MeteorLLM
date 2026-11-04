#include "engine.h"
#include "tokenizer.h"
#include <vita2d.h>
#include <psp2/ctrl.h>
#include <psp2/display.h>
#include <psp2/ime_dialog.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/power.h>
#include <psp2/sysmodule.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

int _newlib_heap_size_user=80*1024*1024;
static const char* root="ux0:data/VitaDesk/MeteorLLM";
static SceUID mutex_id=-1;
static std::atomic<bool> stop_requested{false};
struct Lock{Lock(){sceKernelLockMutex(mutex_id,1,nullptr);}~Lock(){sceKernelUnlockMutex(mutex_id,1);}};
enum Phase{Loading,Ready,Reading,Generating,Failed};
static Phase phase=Loading;
static std::string pending,question,answer,message="Loading the local model",history;
struct Turn { std::string question, answer; };
static std::vector<Turn> completed_turns;
static uint64_t transcript_revision=0;
static bool clear_requested=false;
static uint64_t generation_started=0;
static FILE* log_file=nullptr;
static void log_text(const char*label,const std::string&value){if(log_file){std::fprintf(log_file,"%s %s\n",label,value.c_str());std::fflush(log_file);}}
static int choose(const std::vector<float>&logits,std::mt19937&rng){
 std::vector<int>ids(logits.size());for(size_t i=0;i<ids.size();i++)ids[i]=int(i);
 std::partial_sort(ids.begin(),ids.begin()+20,ids.end(),[&](int a,int b){return logits[a]>logits[b];});ids.resize(20);
 std::vector<double>p;double sum=0;for(int id:ids){double v=std::exp(double(logits[id]-logits[ids[0]]));p.push_back(v);sum+=v;}
 double acc=0;size_t kept=0;while(kept<p.size()&&acc<.8){acc+=p[kept]/sum;++kept;}ids.resize(kept);p.resize(kept);
 for(size_t i=0;i<kept;i++)p[i]=std::exp(double(logits[ids[i]]-logits[ids[0]])/.7);
 std::discrete_distribution<size_t>dist(p.begin(),p.end());return ids[dist(rng)];
}
static std::string read_request(){FILE*f=std::fopen((std::string(root)+"/request.txt").c_str(),"rb");if(!f)return {};char b[4097];size_t n=std::fread(b,1,4096,f);std::fclose(f);return std::string(b,n);}
static int worker(SceSize,void*){
 try{
  log_file=std::fopen((std::string(root)+"/run.log").c_str(),"w");log_text("APP","MeteorLLM offline prototype 00.01");
  uint64_t load_start=sceKernelGetProcessTimeWide();vitaqwen::Engine engine;vitaqwen::Tokenizer tokenizer;engine.load(root);tokenizer.load(std::string(root)+"/tokenizer.bin");engine.cancelled=[](){return bool(stop_requested);};
  log_text("KERNEL_CHECK",std::to_string(engine.validate_quant_kernels())+" values; decode and dot products bit-exact against scalar");
  FILE* diagnostic=std::fopen((std::string(root)+"/matrix-test.txt").c_str(),"r");
  if(diagnostic){std::fclose(diagnostic);engine.benchmark_matrices([](const std::string&s){log_text("MATRIX_TEST",s);});log_text("MATRIX_TEST_DONE","passed");}
  FILE* batch_diagnostic=std::fopen((std::string(root)+"/batch-test.txt").c_str(),"r");if(batch_diagnostic){std::fclose(batch_diagnostic);engine.validate_batch([](const std::string&s){log_text("BATCH_TEST",s);});log_text("BATCH_TEST_DONE","passed");}
  FILE* attention_test=std::fopen((std::string(root)+"/attention-test.txt").c_str(),"r");if(attention_test){std::fclose(attention_test);engine.validate_extended([](const std::string&s){log_text("ATTENTION_CHECK",s);});log_text("ATTENTION_DONE","passed");}
  int attention_mode=0;FILE* attention_config=std::fopen((std::string(root)+"/attention.txt").c_str(),"r");if(attention_config){if(std::fscanf(attention_config,"%d",&attention_mode)!=1)attention_mode=0;std::fclose(attention_config);}if(attention_mode<0||attention_mode>1)throw std::runtime_error("invalid attention mode");engine.set_parallel_attention(attention_mode!=0);log_text("PARALLEL_ATTENTION",std::to_string(attention_mode));
  FILE* extended=std::fopen((std::string(root)+"/extended-test.txt").c_str(),"r");if(extended){std::fclose(extended);engine.benchmark_extended([](const std::string&s){log_text("EXTENDED_KERNEL",s);});engine.validate_extended([](const std::string&s){log_text("EXTENDED_CHECK",s);});engine.benchmark_output([](const std::string&s){log_text("OUTPUT_TEST",s);});log_text("EXTENDED_DONE","passed");}
  int kernel_mode=0;FILE* kernel_config=std::fopen((std::string(root)+"/kernel.txt").c_str(),"r");if(kernel_config){if(std::fscanf(kernel_config,"%d",&kernel_mode)!=1)kernel_mode=0;std::fclose(kernel_config);}engine.set_batch_kernel(kernel_mode);log_text("BATCH_KERNEL",std::to_string(kernel_mode));
  int output_rows=64;FILE* output_config=std::fopen((std::string(root)+"/output-rows.txt").c_str(),"r");if(output_config){if(std::fscanf(output_config,"%d",&output_rows)!=1)output_rows=64;std::fclose(output_config);}engine.set_output_rows(output_rows);log_text("OUTPUT_ROWS",std::to_string(output_rows));
  int batch_size=4;FILE* batch_config=std::fopen((std::string(root)+"/batch.txt").c_str(),"r");if(batch_config){if(std::fscanf(batch_config,"%d",&batch_size)!=1)batch_size=1;std::fclose(batch_config);}if(batch_size<1||batch_size>16)throw std::runtime_error("invalid batch size");log_text("BATCH_SIZE",std::to_string(batch_size));
  int matrix_workers=3;FILE* config=std::fopen((std::string(root)+"/workers.txt").c_str(),"r");if(config){if(std::fscanf(config,"%d",&matrix_workers)!=1)matrix_workers=1;std::fclose(config);}engine.set_workers(matrix_workers);log_text("WORKERS",std::to_string(matrix_workers));
  SceKernelFreeMemorySizeInfo mem{};mem.size=sizeof(mem);sceKernelGetFreeMemorySize(&mem);
  log_text("LOAD",std::to_string((sceKernelGetProcessTimeWide()-load_start)/1000)+" ms; resident="+std::to_string(engine.resident_bytes())+" free_user="+std::to_string(mem.size_user)+" cpu="+std::to_string(scePowerGetArmClockFrequency()));
  {Lock lock;phase=Ready;message="Ready when you are";pending=read_request();}
  for(;;){
   std::string input,prior;
   {Lock lock;if(clear_requested){history.clear();question.clear();answer.clear();completed_turns.clear();++transcript_revision;clear_requested=false;}if(!pending.empty()){input=pending;pending.clear();prior=history;if(!question.empty())completed_turns.push_back({question,answer});question=input;answer.clear();++transcript_revision;phase=Reading;message="Reading your question";generation_started=sceKernelGetProcessTimeWide();}}
   if(input.empty()){sceKernelDelayThread(20000);continue;}
   stop_requested=false;log_text("PROMPT",input);std::string response;
   try{
    std::string prompt="<|im_start|>system\nYou are a helpful assistant. Follow the instructions and answer concisely. If you do not know, say so.<|im_end|>\n"+prior+"<|im_start|>user\n"+input+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    auto ids=tokenizer.encode(prompt);if(ids.empty()||ids.size()+16>=size_t(engine.config().context))throw std::runtime_error("This conversation is full. Start a new chat.");
    log_text("ENCODED",std::to_string(ids.size())+" tokens; elapsed_us="+std::to_string(sceKernelGetProcessTimeWide()-generation_started));
    engine.progress=[&](int token,int layer,const char*stage){
     if(token==0||layer==-1)log_text("PROGRESS","token="+std::to_string(token)+" layer="+std::to_string(layer)+" stage="+stage+" elapsed_us="+std::to_string(sceKernelGetProcessTimeWide()-generation_started));
     if(layer>=0){Lock lock;message=(phase==Reading?"Reading "+std::to_string(token+1)+"/"+std::to_string(ids.size()):"Generating token "+std::to_string(token-int(ids.size())+2))+"  layer "+std::to_string(layer+1)+"/"+std::to_string(engine.config().layers);}
     sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
    };
    engine.reset();const std::vector<float>*scores=nullptr;int pos=0;
    while(pos<int(ids.size())){if(stop_requested)break;int n=std::min(batch_size,int(ids.size())-pos);auto start=sceKernelGetProcessTimeWide();scores=&engine.forward_batch(std::vector<int>(ids.begin()+pos,ids.begin()+pos+n),pos,pos+n==int(ids.size()));pos+=n;log_text("PREFILL","position="+std::to_string(pos)+" batch="+std::to_string(n)+" elapsed_us="+std::to_string(sceKernelGetProcessTimeWide()-start)+" "+engine.stage_metrics());sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);}
    std::mt19937 rng(1234);int generated=0;bool eos=false;uint64_t first_token=0;
    if(!stop_requested){Lock lock;phase=Generating;message="Generating on your Vita";}
    while(!stop_requested&&scores&&pos<engine.config().context&&generated<128){
     int next=choose(*scores,rng);if(next==engine.config().eos||next==151643){eos=true;break;}
     response+=tokenizer.decode(next);generated++;if(!first_token)first_token=sceKernelGetProcessTimeWide();
     {Lock lock;answer=response;++transcript_revision;}
     log_text("TOKEN",std::to_string(next));if(pos+1>=engine.config().context||generated==128)break;scores=&engine.forward(next,pos++);sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
    }
    uint64_t finished=sceKernelGetProcessTimeWide();log_text("ANSWER",response);log_text("METRICS","prompt_tokens="+std::to_string(ids.size())+" generated="+std::to_string(generated)+" eos="+std::to_string(eos)+" stopped="+std::to_string(bool(stop_requested))+" elapsed_us="+std::to_string(finished-generation_started)+" first_token_us="+std::to_string(first_token?first_token-generation_started:0)+" embedding_read_bytes="+std::to_string(engine.embedding_read_bytes)+" output_read_us="+std::to_string(engine.output_read_us)+" output_compute_us="+std::to_string(engine.output_compute_us)+" "+engine.stage_metrics());
    {Lock lock;if(!response.empty())history=prior+"<|im_start|>user\n"+input+"<|im_end|>\n<|im_start|>assistant\n"+response+"<|im_end|>\n";phase=Ready;message=stop_requested?"Stopped":eos?"Ready when you are":"Response limit reached";}
   }catch(const std::exception&e){log_text("GENERATION_ERROR",std::string(e.what())+" elapsed_us="+std::to_string(sceKernelGetProcessTimeWide()-generation_started));Lock lock;phase=Ready;message=e.what();}
  }
 }catch(const std::exception&e){log_text("LOAD_ERROR",e.what());Lock lock;phase=Failed;message=e.what();}
 return 0;
}
static std::string utf8(const SceWChar16*in){std::string out;for(size_t i=0;in[i];i++){uint32_t cp=in[i];if(cp>=0xd800&&cp<=0xdbff&&in[i+1]>=0xdc00&&in[i+1]<=0xdfff){cp=0x10000+((cp-0xd800)<<10)+(in[++i]-0xdc00);}if(cp<128)out+=char(cp);else if(cp<2048){out+=char(0xc0|(cp>>6));out+=char(0x80|(cp&63));}else if(cp<65536){out+=char(0xe0|(cp>>12));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}else{out+=char(0xf0|(cp>>18));out+=char(0x80|((cp>>12)&63));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}}return out;}
static std::vector<std::string>wrap(vita2d_pgf*font,const std::string&text,int width){
 std::vector<std::string>lines;std::istringstream paragraphs(text);std::string paragraph;
 while(std::getline(paragraphs,paragraph)){std::istringstream words(paragraph);std::string word,line;while(words>>word){std::string next=line.empty()?word:line+" "+word;if(!line.empty()&&vita2d_pgf_text_width(font,1.0f,next.c_str())>width){lines.push_back(line);line=word;}else line=next;}lines.push_back(line);}
 return lines;
}
int main(){
 mutex_id=sceKernelCreateMutex("meteor_ui",0,0,nullptr);if(mutex_id<0)return 1;sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG);sceSysmoduleLoadModule(SCE_SYSMODULE_IME);
 if(vita2d_init()<0)return 2;
 vita2d_set_clear_color(RGBA8(8,12,22,255));vita2d_pgf*font=vita2d_load_default_pgf();if(!font)return 3;
 SceUID thread=sceKernelCreateThread("meteor_inference",worker,0x10000100,256*1024,0,0,nullptr);if(thread<0||sceKernelStartThread(thread,0,nullptr)<0){Lock lock;phase=Failed;message="Could not start the model worker";}
 uint32_t last=0;bool ime=false;SceWChar16 input[1025]={0},empty[1]={0};static const SceWChar16 title[]={'A','s','k',' ','M','e','t','e','o','r','L','L','M',0};
 struct ChatLine { std::string text; bool label; };
 std::vector<ChatLine> lines;
 constexpr int visible_lines=15;
 uint64_t seen_revision=0,next_scroll=0;
 int scroll=0,last_direction=0;
 bool follow_latest=true;
 const unsigned amber=RGBA8(255,177,105,255),light=RGBA8(236,239,247,255),muted=RGBA8(147,160,182,255),panel=RGBA8(17,24,39,255);
 for(;;){
  Phase state;
  std::string status;
  uint64_t started;
  std::vector<Turn> snapshot;
  bool changed=false;
  {
   Lock lock;
   state=phase;status=message;started=generation_started;
   if(seen_revision!=transcript_revision){
    seen_revision=transcript_revision;
    snapshot=completed_turns;
    if(!question.empty())snapshot.push_back({question,answer});
    changed=true;
   }
  }
  if(changed){
   lines.clear();
   for(const auto& turn:snapshot){
    if(!lines.empty())lines.push_back({"",false});
    lines.push_back({"YOU",true});
    for(auto& line:wrap(font,turn.question,836))lines.push_back({std::move(line),false});
    if(!turn.answer.empty()){
     lines.push_back({"METEOR",true});
     for(auto& line:wrap(font,turn.answer,836))lines.push_back({std::move(line),false});
    }
   }
   if(lines.empty()){scroll=0;follow_latest=true;}
  }
  const int max_scroll=std::max(0,int(lines.size())-visible_lines);
  scroll=follow_latest?max_scroll:std::min(scroll,max_scroll);
  SceCtrlData pad{};sceCtrlPeekBufferPositive(0,&pad,1);uint32_t pressed=pad.buttons&~last;last=pad.buttons;
  if(!ime){
   if((pressed&SCE_CTRL_CIRCLE)&&(state==Reading||state==Generating))stop_requested=true;
   if((pressed&SCE_CTRL_TRIANGLE)&&state==Ready){Lock lock;clear_requested=true;message="New chat";}
   const int direction=((pad.buttons&SCE_CTRL_DOWN)!=0)-((pad.buttons&SCE_CTRL_UP)!=0);
   const auto now=sceKernelGetProcessTimeWide();
   if(direction&&(direction!=last_direction||now>=next_scroll)){
    scroll=std::clamp(scroll+direction,0,max_scroll);
    follow_latest=scroll==max_scroll;
    next_scroll=now+(direction==last_direction?70000:300000);
   }
   last_direction=direction;
   if((pressed&SCE_CTRL_CROSS)&&state==Ready){SceImeDialogParam param;sceImeDialogParamInit(&param);param.supportedLanguages=SCE_IME_LANGUAGE_ENGLISH;param.type=SCE_IME_TYPE_BASIC_LATIN;param.title=title;param.maxTextLength=1024;param.initialText=empty;param.inputTextBuffer=input;param.dialogMode=SCE_IME_DIALOG_DIALOG_MODE_WITH_CANCEL;if(sceImeDialogInit(&param)>=0)ime=true;}
  }else if(sceImeDialogGetStatus()==SCE_COMMON_DIALOG_STATUS_FINISHED){SceImeDialogResult result{};sceImeDialogGetResult(&result);sceImeDialogTerm();ime=false;if(result.button==SCE_IME_DIALOG_BUTTON_ENTER){std::string text=utf8(input);if(!text.empty()){Lock lock;pending=text;}}}
  vita2d_start_drawing();vita2d_clear_screen();
  // Native meteor mark: warm nucleus and three tapering trails.
  for(int i=0;i<3;i++){vita2d_draw_line(25+i*6,16,49+i*5,45,RGBA8(111+i*36,66+i*21,58+i*12,255));}
  vita2d_draw_fill_circle(57,49,12,RGBA8(121,65,42,255));vita2d_draw_fill_circle(57,49,8,amber);vita2d_draw_fill_circle(59,47,3,RGBA8(255,230,185,255));
  vita2d_pgf_draw_text(font,84,57,light,1.5f,"MeteorLLM");vita2d_pgf_draw_text(font,85,79,muted,.65f,"OFFLINE INTELLIGENCE");
  vita2d_draw_rectangle(786,29,145,37,panel);vita2d_draw_fill_circle(804,47,3,amber);vita2d_pgf_draw_text(font,818,53,amber,.8f,"ON DEVICE");
  vita2d_draw_line(28,96,932,96,RGBA8(37,47,66,255));
  if(lines.empty()){
   vita2d_pgf_draw_text(font,54,188,light,1.4f,"A little more space to think.");
   vita2d_pgf_draw_text(font,54,229,muted,1.0f,"Ask a question, explore an idea, or write something new.");
   vita2d_pgf_draw_text(font,54,260,muted,1.0f,"Everything runs right here on your Vita.");
   vita2d_draw_line(754,190,842,286,RGBA8(55,39,43,255));vita2d_draw_line(774,190,854,279,RGBA8(118,68,48,255));vita2d_draw_fill_circle(860,286,10,amber);
  }else{
   vita2d_draw_rectangle(28,110,904,366,panel);
   for(int i=0;i<visible_lines&&scroll+i<int(lines.size());i++){
    const auto& line=lines[scroll+i];
    vita2d_pgf_draw_text(font,48,135+23*i,line.label?amber:light,line.label?.75f:1.0f,line.text.c_str());
   }
   if(max_scroll){
    const float thumb=std::max(14.f,350.f*visible_lines/lines.size());
    vita2d_draw_rectangle(918,118+(350-thumb)*scroll/max_scroll,4,thumb,muted);
   }
  }
  std::string bottom=status;if((state==Reading||state==Generating)&&started){uint64_t sec=(sceKernelGetProcessTimeWide()-started)/1000000;char time[32];std::snprintf(time,sizeof(time),"  %llu:%02llu",static_cast<unsigned long long>(sec/60),static_cast<unsigned long long>(sec%60));bottom+=time;}
  vita2d_pgf_draw_text(font,30,501,state==Failed?RGBA8(255,120,112,255):muted,.8f,bottom.c_str());
  vita2d_pgf_draw_text(font,30,531,light,.75f,"X  Ask     Triangle  New chat     Circle  Stop");vita2d_pgf_draw_text(font,754,531,muted,.7f,"Up / Down  Scroll");
  vita2d_end_drawing();vita2d_common_dialog_update();vita2d_swap_buffers();sceDisplayWaitVblankStart();
 }
}
