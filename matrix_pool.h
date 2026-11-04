// Authored for MeteorLLM. Jobs partition complete rows; no shared scratch.
#pragma once
#include <atomic>
#include <functional>
#include <stdexcept>
#ifdef __vita__
#include <psp2/kernel/threadmgr.h>
#else
#include <thread>
#include <mutex>
#include <condition_variable>
#endif
namespace vitaqwen {
class MatrixPool {
public:
 MatrixPool()=default;
 ~MatrixPool(){shutdown();}
 MatrixPool(const MatrixPool&)=delete;
 void run(int count,int rows,const std::function<void(int,int,int)>& fn){
  if(count<1||count>3)throw std::runtime_error("invalid matrix worker count");
  if(count==1){fn(0,rows,0);return;}
  start(); job=fn; total=rows; workers=count;
  for(int i=0;i<count-1;i++)wake(i);
  fn(rows*(count-1)/count,rows,count-1);
  for(int i=0;i<count-1;i++)wait_done(i);
  job={};
 }
private:
 std::function<void(int,int,int)> job;
 int total=0,workers=1,started=0;
 std::atomic<bool> stop{false};
#ifdef __vita__
 int threads[2]={-1,-1},ready[2]={-1,-1},done[2]={-1,-1};
 struct Arg{MatrixPool* pool;int index;};
 static int entry(SceSize,void* p){auto a=*static_cast<Arg*>(p);a.pool->loop(a.index);return 0;}
 void wake(int i){sceKernelSignalSema(ready[i],1);}
 void wait_done(int i){sceKernelWaitSema(done[i],1,nullptr);}
 void loop(int i){for(;;){sceKernelWaitSema(ready[i],1,nullptr);if(stop.load(std::memory_order_acquire))return;job(total*i/workers,total*(i+1)/workers,i);sceKernelSignalSema(done[i],1);}}
 void start(){if(started==2)return;stop=false;try{for(int i=0;i<2;i++){
  ready[i]=sceKernelCreateSema("meteor_work",0,0,1,nullptr);
  done[i]=sceKernelCreateSema("meteor_done",0,0,1,nullptr);
  if(ready[i]<0||done[i]<0)throw std::runtime_error("matrix semaphore creation failed");
  threads[i]=sceKernelCreateThread("meteor_matrix",entry,0x10000100,64*1024,0,0,nullptr);
  if(threads[i]<0)throw std::runtime_error("matrix thread creation failed");
  Arg a{this,i};if(sceKernelStartThread(threads[i],sizeof(a),&a)<0)throw std::runtime_error("matrix thread start failed");++started;
 }}catch(...){shutdown();throw;}}
 void shutdown(){stop.store(true,std::memory_order_release);for(int i=0;i<started;i++)wake(i);for(int i=0;i<started;i++)sceKernelWaitThreadEnd(threads[i],nullptr,nullptr);for(int i=0;i<2;i++){if(threads[i]>=0)sceKernelDeleteThread(threads[i]);if(ready[i]>=0)sceKernelDeleteSema(ready[i]);if(done[i]>=0)sceKernelDeleteSema(done[i]);threads[i]=ready[i]=done[i]=-1;}started=0;}
#else
 std::thread threads[2];std::mutex mutex;std::condition_variable cv;
 bool ready[2]={false,false},done[2]={false,false};
 void wake(int i){{std::lock_guard<std::mutex> lock(mutex);ready[i]=true;}cv.notify_all();}
 void wait_done(int i){std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return done[i];});done[i]=false;}
 void loop(int i){for(;;){{std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return ready[i];});ready[i]=false;if(stop)return;}job(total*i/workers,total*(i+1)/workers,i);{std::lock_guard<std::mutex> lock(mutex);done[i]=true;}cv.notify_all();}}
 void start(){if(started==2)return;stop=false;for(int i=0;i<2;i++){ready[i]=false;done[i]=false;}try{for(int i=0;i<2;i++){threads[i]=std::thread([this,i]{loop(i);});++started;}}catch(...){shutdown();throw;}}
 void shutdown(){stop=true;for(int i=0;i<started;i++)wake(i);for(int i=0;i<started;i++)threads[i].join();started=0;}
#endif
};
}
