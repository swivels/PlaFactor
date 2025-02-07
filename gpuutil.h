//
// Basic GPU setup and API's

//
// This code should be compiled with -fopenmp under gcc and
// equivalent for other compilers.
//

// Important environment variables:
//   export OMP_NUM_THREADS=<num>           # number of openmp threads to use
//   export CUDA_VISIBLE_DEVICES=<devlist>  # comma separated device numbers
//                                          # with 1 device it will be <devlist>=0
//                                          # with 4 devices it could be 0,3
//                                          # which leaves out 1, and 2
#ifndef GPUUTIL_H
#define GPUUTIL_H

#ifdef __NVCC__
#define NVCC
#endif

#include <cmath>
#include <cstddef> // for offsetof
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include "omp.h"
#ifdef NVCC
#include <cuda.h>
#include <cuda_runtime.h>

// Used to control where functions and variables are "built"
// This means a function or variable is available on CPU and GPU
#define NVCC_BOTH __host__ __device__
// This means only on the GPU
#define NVCC_DEVICE __device__
#endif

// Fatal error message printer - will exit.
void GpuErr(const char *fmt, ...);

// GpuInit will set these up
extern int GpuNumCpuThreads; // number of OMP threads
extern int *GpuThreadDev;    // gives device for OMP thread
extern int GpuNumDevices;    // number of GPU devices
extern int GpuBlockSize;     // number of threads in a block for kernal launch
extern int GpuBlockShift;    // GpuBlockSize == (1 << GpuBlockShift) is a power of 2

#ifdef NVCC
// Streams let multiple threads share a GPU - can improve throughput
extern cudaStream_t *GpuThreadStream;  // map thread to GPU stream
#endif

// Call this at startup so functions/tables in this file are set up.
void GpuInit(int argc, char **argv);

#ifdef NVCC
#define GpuCheckErrors(ans_) { \
    cudaError_t code_ = (ans_); \
    if(code_ != cudaSuccess) GpuErr("Gpuassert: %s, code=%d\n", cudaGetErrorString(code_), (int)code_); \
}
#define GpuCheckErrorsMsg(ans_, msg_) { \
    cudaError_t code_ = (ans_); \
    if(code_ != cudaSuccess) GpuErr("Gpuassert: %s ,code=%d: %s\n", cudaGetErrorString(code_), (int)code_, msg_); \
}

// default is to determine device for thread, but can be manually controlled.
void GpuSetThreadDevice(int tid = -1);

// Pinned memory is CPU memory suitable for transmitting to GPUs because
// it is page locked.   An async DMA transfer can count on it staying put.
// Only use for buffers used for transers.
void GpuMallocPinned(void **p, size_t sz);
void GpuFreePinned(void *p);

// Allocate memory on the device attached to current thread (GpuThreadDev)
void GpuMallocDevice(void **d_p, size_t sz);
void GpuFreeDevice(void *d_p);

void GpuMemcpyHostToDevice(void *d_mem, const void *h_mem, size_t sz, cudaStream_t astream = 0);
void GpuMemcpyDeviceToHost(void *h_mem, const void *d_mem, size_t sz, cudaStream_t astream = 0);

void GpuGetDevice(int &dev);
int GpuGetDevice();


// Support for launching computations on large arrays with threads + multiple GPUs
#define GPUSETUPTHREAD(tid_, stream_) \
	int tid_ = omp_get_thread_num(); \
	GpuSetThreadDevice(tid_); \
	cudaStream_t stream_ = GpuThreadStream[tid_]; 


static inline int GpuThreads2BlockCount(size_t nthreads) {
	// >> is divide by GpuBlockSize
	return (int)((nthreads + GpuBlockSize - 1) >> GpuBlockShift);  
}

//
// Call from OMP thread to figure out division of labor.
//
// Given cnt GPU threads to launch, figure out the starting
// block and block count that this OMP thread will be responsible for.
//
// Input:
//    cnt: number of GPU threads to launch across OMP threads
// Output:
//    kblks:  number of blocks for kernel launch from this OMP thread
//    toff
//
inline static void GpuThreadDelegate(size_t cnt, int &tid, cudaStream_t &s,  int &kblks, size_t &toff, size_t &rndtcnt, size_t &tcnt) {
	tid = omp_get_thread_num();  // get id of current OMP thread
	GpuSetThreadDevice(tid);
	s = GpuThreadStream[tid];  // keeps threads sharing a GPU from interfering
	int totalblocks = GpuThreads2BlockCount(cnt);  // rounded up 
	int base = totalblocks / GpuNumCpuThreads;     // base number of blocks
	int leftover = totalblocks % GpuNumCpuThreads; // extra blocks, add one to first leftover threads
	int sblk = min(tid, leftover) + base * tid;          // starting block
	kblks = base + ((tid < leftover) ? 1 : 0);     // number of blocks to launch for kernel
	int endb = sblk + kblks;   // endb is block after this thread's group

	// Now compute the GPU thread numbers handled by this OMP thread
	size_t tend = std::min(cnt, (size_t)endb * GpuBlockSize); // index of first GPU thread after this OMP threads group
	toff = sblk * GpuBlockSize;   // First GPU thread handled by this OMP thread
	rndtcnt = kblks * GpuBlockSize; // used for alloc on GPU - rounded up to block boundary
	tcnt = rndtcnt;                 // used for copy and cnt for kernal - exact
	if(toff + tcnt > cnt) tcnt = cnt - toff; // last block is trimmed to actual cnt
}

// Information about clones on a per OMP thread basis
struct GpuCloneInfo_t { 
	void *gpup;  // pointer to buffer on GPU
	size_t data; // extra info about buffer
};

struct GpuCloneRecord_t {
	GpuCloneRecord_t *gnext; // next in hash bin list
	const void *cpup;        // cpu buffer being cloned
	size_t size;             // size of cpu buffer
	bool   clonethread;      // true -> clone/thread, false -> clone/device
	GpuCloneInfo_t *info;    // [tid], one entry per thread

	GpuCloneRecord_t();
	~GpuCloneRecord_t();
};

// A table of these are used for patching pointers inside
// struct being cloned.   Clone things pointed to first, then
// clone top structure.
struct GpuClonePatch_t {
	//void *cpup;  not needed      // pointer to cpu buffer that was cloned, nullptr to zap pointer in clone
	int offset;    // use offsetof(type, field)
};
#define GPUDECLPATCHES(maxpatches) \
	unsigned _patchcnt = 0; \
	GpuClonePatch_t _patches[maxpatches+2];

#define ADDPATCH(cpup_, field_) {  \
	unsigned pos = _patchcnt++; \
	GpuClonePatch_t *p = &_patches[pos]; \
	p->offset = offsetof(decltype(*cpup_), field_); \
}

#define ENDPATCH() { \
	unsigned pos = _patchcnt++; \
	GpuClonePatch_t *p = &_patches[pos]; \
	p->offset = -1; \
}

//	p->offset = ((char *)&cpup_->field_) - (char *)cpup_); \

// Clone once per device for read only data shared across threads
// TODO:  might want to put in "constant" memory
void GpuCloneForDevices(void *cpup, size_t size, bool update=true, GpuClonePatch_t *patches = nullptr);
// If using streams, cpup should be allocated with
//   GpuMallocPinned so physical address doesn't change during
//   async transfer.
void GpuCloneForThreads(void *cpup, size_t size, bool update=false, GpuClonePatch_t *patches = nullptr);
// Find clone this thread should use.
void *GpuFindCloneThread(void *cpup, int tid = -1);
void *GpuFindCloneDev(void *cpup, int dev);

// free all associated clones
void GpuFreeClones(void *cpup); 

#endif

// File guard endif
#endif
