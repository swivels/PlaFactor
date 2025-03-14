/*
Copyright ©  2025   The Regents of the University of California
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
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
#include <atomic>
#include "gpuutil.h"

void GpuErr(const char *fmt,...) {
	int tid = omp_get_thread_num();
	va_list args;
	printf("tid@%d - ", tid);
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	printf("\n");
	exit(1);
}

NVCC_BOTH void OnGpuErr(const char *fmt) {
	printf("%s\n", fmt);
#ifdef __CUDA_ARCH__
	printf("Error on GPU\n");
	asm("trap;");
#else
	printf("Error on CPU\n");
	exit(1);
#endif
}

void GpuCheckKernelLaunch(const char *msg) {
	cudaError_t err = cudaGetLastError(); //put after every kernel launch
	if(cudaSuccess != err) {
		GpuErr("Kernel %s(%d): %s\n",msg, static_cast<int>(err), cudaGetErrorString(err));
	}
}


int GpuNumCpuThreads = 0;
int *GpuThreadDevice = nullptr;
int GpuNumDevices = 0;
int GpuBlockSize = 0;
int GpuBlockShift = 0;  // will have  GpuBlockSize == (1 << GpuBlockShift)
cudaDeviceProp *GpuDeviceProp = nullptr;
//
// Streams keep each's threads actions on a GPU separate from
// other threads actions.   Multiple threads will share one GPU.
cudaStream_t *GpuThreadStream = nullptr;

// Set current device - GPU
void GpuSetDevice(int dev) {
	GpuCheckErrors(cudaSetDevice(dev));
}

// Get current device - GPU
int GpuGetDevice() {
	int dev;
	GpuCheckErrors(cudaGetDevice(&dev));
	return dev;
}
void GpuGetDevice(int &dev) {
	GpuCheckErrors(cudaGetDevice(&dev));
}

// Call before using the GPU
void GpuInit(int argc, char **argv) {
	GpuNumCpuThreads = omp_get_max_threads();
	GpuThreadDevice = new int[GpuNumCpuThreads]; // figure out which gpu a thread uses.
	GpuThreadStream = new cudaStream_t[GpuNumCpuThreads];

	// See if user is picking GPU's for computation.
	const char *s = getenv("CUDA_VISIBLE_DEVICES");

	// Note, allmost all cuda API functions return an error code
	// GpuCheckErrors make sure the call worked.   Try to do this
	// even when you can't see how it would fail.
	GpuCheckErrors(cudaGetDeviceCount(&GpuNumDevices));
	if(GpuNumDevices <= 0) GpuErr("Can't find GPU");
	GpuDeviceProp = new cudaDeviceProp[GpuNumDevices];
	for(int dev = 0; dev < GpuNumDevices; dev++) {
		GpuCheckErrors(cudaGetDeviceProperties(&GpuDeviceProp[dev], dev));
		int bs = GpuDeviceProp[dev].maxThreadsPerBlock;
		if(GpuBlockSize != 0 && bs != GpuBlockSize) GpuErr("Inconsistant block sizes on multiple GPUs");
		GpuBlockSize = bs;
	}
	// GpuBlockSize should be power of 2
	int bshift = 1;
	while( (1 << bshift) < GpuBlockSize ) bshift++;
	if ( (1 << bshift) != GpuBlockSize) GpuErr("Block size is not a power of 2???");
	GpuBlockShift = bshift;

	// do a very simple assignment of threads to devices.
	int savedev;
	GpuCheckErrors(cudaGetDevice(&savedev)); // save orig thread
	for(int tid = 0; tid < GpuNumCpuThreads; tid++) { // loop over OMP threads
		int dev = tid % GpuNumDevices;       // get device for OMP thread tid
		GpuThreadDevice[tid] = dev;          // save in mapping table  tid->dev
		GpuCheckErrors(cudaSetDevice(dev));  // set current device for stream creation
		GpuCheckErrors(cudaStreamCreate(&GpuThreadStream[tid])); // create the stream (virtual GPU) for thread tid
	}
	GpuCheckErrors(cudaSetDevice(savedev));  // restore original thread
}

// "Pinned" memory is CPU memory suitable for transmitting to GPUs because
// it is page locked AKA "pinned".   An async DMA transfer can count on it staying put.
// Only use for buffers used for transers.
void GpuMallocPinned(void **p, size_t sz) {
	// cuda calls it "Host" memory and then describes it as pinned
	GpuCheckErrors(cudaMallocHost(p, sz));
}

//
// Managed memory will page back and forth between GPU/CPU
// Not as fast, but can be simpler for some things.
// Docs not too clear for multiple GPUs
//
void GpuMallocManaged(void **p, size_t sz) {
	GpuCheckErrors(cudaMallocManaged(p, sz));
}

// Allocate memory on the device attached to current thread (GpuThreadDevice)
void GpuMallocDevice(void **p, size_t sz) {
	GpuCheckErrors(cudaMalloc(p, sz));
}

void GpuFreeDevice(void *p) {
	GpuCheckErrors(cudaFree(p));
}

void GpuFreePinned(void *p) {
	GpuCheckErrors(cudaFreeHost(p));
}

// Copy from host to device,  like memcpy first arg is destination.
void GpuMemcpyHostToDevice(void *d_mem, const void *h_mem, size_t sz, cudaStream_t astream) {
	cudaError_t code;
	if(astream) {
		code = cudaMemcpyAsync(d_mem, h_mem, sz, cudaMemcpyHostToDevice, astream);
	} else {
		// Memory does not need to be pinned/managed in this case
		// Often used to initialize data structures from thread 0
		code = cudaMemcpy(d_mem, h_mem, sz, cudaMemcpyHostToDevice);
	}
	GpuCheckErrors(code);
}

void GpuMemcpyDeviceToHost(void *h_mem, const void *d_mem, size_t sz, cudaStream_t astream) {
	cudaError_t code;
	if(astream) {
		code = cudaMemcpyAsync(h_mem, d_mem, sz, cudaMemcpyDeviceToHost, astream);
	} else {
		// Memory does not need to be pinned/managed in this case
		// Often used to initialize data structures from thread 0
		code = cudaMemcpy(h_mem, d_mem, sz, cudaMemcpyDeviceToHost);
	}
	GpuCheckErrors(code);
}

// Call when in OMP thread to set up
// #pragma omp parallel
//    {
//        GpuSetThreadDevice(); // done once per thread
// #pragma omp for
//        for(int i = 0; i < cnt; i++) {
//            Do work.
//        }
//    }
void GpuSetThreadDevice(int tid/* = -1*/) {
	int xtid;
	if(tid < 0) {
		xtid = omp_get_thread_num();
	} else {
		xtid = tid;
	}
	GpuCheckErrors(cudaSetDevice(GpuThreadDevice[xtid]));
}

// Cloning support

#define CLONEHASHSIZE (1u << 14u)
static GpuCloneRecord_t *clonehashtab[CLONEHASHSIZE] = {0};
static std::atomic_flag clonereglock = ATOMIC_FLAG_INIT; // init to not set
static int clonecount = 0;

// hash a pointer into clonehashtab index
static inline unsigned clonehash(const void *p) {
    size_t pi = (size_t)p;
    return ((pi >> 4u) + (pi >> 18u)) & (CLONEHASHSIZE-1);
}

// associated with one cpu pointer
GpuCloneRecord_t::GpuCloneRecord_t() {
	gnext = nullptr;
	cpup = nullptr;
	info = new GpuCloneInfo_t[GpuNumCpuThreads]; // one record per possible thread
	for(int i = 0; i < GpuNumCpuThreads; i++) {
		info[i].gpup = nullptr;
		info[i].data = 0;
	}
}

GpuCloneRecord_t::~GpuCloneRecord_t() {
	delete[] info;
}

static GpuCloneRecord_t *getclonerecord(const void *cpup) {
	unsigned h = clonehash(cpup);
	GpuCloneRecord_t *crp;
	for(crp = clonehashtab[h]; crp; crp=crp->gnext) {
		if(crp->cpup == cpup) {
			return crp;
		}
	}
	return nullptr;
}

// Get clone, record, return nullptr if not found
static GpuCloneRecord_t *getclonerecord(const void *cpup, bool clonethread) {
	unsigned h = clonehash(cpup);
	GpuCloneRecord_t *crp;
	for(crp = clonehashtab[h]; crp; crp=crp->gnext) {
		if(crp->cpup == cpup) {
			if(!clonethread && crp->clonethread)
				GpuErr("getclonerecord: dev/thread mismatch");
			return crp;
		}
	}
	return nullptr;
}

// Get or if needed make a clone record for cpu buffer pointer cpup
static GpuCloneRecord_t *getclonerecord(const void *cpup, size_t size, bool clonethread) {
	unsigned h = clonehash(cpup);
	GpuCloneRecord_t *crp;
	for(crp = clonehashtab[h]; crp; crp=crp->gnext) {
		if(crp->cpup == cpup) {
			if(crp->size != size) 
				GpuErr("Clone size mismatch");
			if(!clonethread && crp->clonethread)
				GpuErr("getclonerecord: dev/thread mismatch");
			return crp;
		}
	}
	// have to make one
    crp = new GpuCloneRecord_t;
    crp->cpup = cpup;
	crp->size = size;
	crp->clonethread = clonethread;
    // lock before inserting
    while(clonereglock.test_and_set(std::memory_order_acquire))
        ; // spin
    crp->gnext = clonehashtab[h];
    clonehashtab[h] = crp;
    clonecount++;
    // unlock
    clonereglock.clear(std::memory_order_release);
    return crp;
}

//
// Each CPU buffer may have either one clone per thread
// or one per device (shared by threads sharing that device)
// Constant tables shared in calculations fit the per device option.
// Buffers to send calculation inputs or retrieve calculation results
// will usually be on a thread basis.
// supply optional arg omp thread id tid if you already have it
void *GpuFindCloneThread(void *cpup, int tid) {
	GpuCloneRecord_t *crp = getclonerecord(cpup, true);
	if(!crp) GpuErr("Missing clone");
	if(tid < 0) tid = omp_get_thread_num();
	return crp->info[tid].gpup;
}

//
// Some data is cloned once per device like constant tables
// used across all threads.
//
void *GpuFindCloneDevice(void *cpup, int dev) {
	// false is for not thread based.
	GpuCloneRecord_t *crp = getclonerecord(cpup, false);
	if(crp->clonethread) GpuErr("GpuFindCloneDevice with thread clone");
	GpuCloneInfo_t *info = crp->info;
	for(int tid = 0; tid <= GpuNumCpuThreads; tid++) {
		int tdev = GpuThreadDevice[tid];
		if(tdev == dev) {
			return info[tdev].gpup;
		}
	}
	return nullptr;
}

static int GpuClonePatchEndInt = 0;
// pointer value used to mark end of patches
void *GpuClonePatchEnd = (void *)&GpuClonePatchEndInt;

//
// Each thread gets a clone.
// Do not update by default.  We will usually perform copies
// immediately before or after kernel launches.
//
// update false with patches doens't make sense.
//
//if update is false, still allocate but don't copy data
//clone subfields before cloning overarching main struct/class and then patch the subfields in
void GpuCloneForThreads(void *cpup, size_t size, bool update, GpuClonePatch_t *patches) {
	int savedev = GpuGetDevice();
	GpuCloneRecord_t *crp = getclonerecord(cpup, size, true);
	char *pbuf = nullptr;
	if (update && patches) {
		pbuf = new char[size];
		memcpy(pbuf, cpup, size);
	}
	for(int tid = 0; tid < GpuNumCpuThreads; tid++) {
		GpuSetThreadDevice(tid);
		void *gpup;
		GpuMallocDevice(&gpup, size);
		crp->info[tid].gpup = gpup;
		crp->info[tid].data = 0;
		if(update) {
			if(patches) {
				// fill buffer pbuf with *cpup + patches
				void *fp;
				for(GpuClonePatch_t *p = &patches[0]; p->offset>=0; p++) {
					memcpy((void*)&fp, (char*)cpup + p->offset, sizeof(void *));
					void *dpp = GpuFindCloneThread(fp, tid);
					if (!dpp)
						GpuErr("Missing field clone when patching");
					memcpy(pbuf + p->offset, (void *)&dpp, sizeof(void *));
				}
				GpuMemcpyHostToDevice(gpup, pbuf, size);
			} else {
				// no patches, send original data
				GpuMemcpyHostToDevice(gpup, cpup, size);
			}
		}
	}
	GpuSetDevice(savedev);
	if(pbuf) delete[] pbuf;
}

//
// Make a clone per device.   By default, update clone from cpu
// immediately.   This is the usual thing to do for 
// shared (between threads) tables.   You update once at creation.
//
// update false with patches doesn't make sense.
//
void GpuCloneForDevices(void *cpup, size_t size, bool update, GpuClonePatch_t *patches) {
	int savedev = GpuGetDevice();
	GpuCloneRecord_t *crp = getclonerecord(cpup, size, false);
	char *pbuf = nullptr;
	if (update && patches) {
		pbuf = new char[size];
		memcpy(pbuf, cpup, size);
	}
	void **dclones = new void *[GpuNumDevices];
	for(int dev = 0; dev < GpuNumDevices; dev++) {
		GpuSetDevice(dev);
		GpuMallocDevice(&dclones[dev], size);
		if(update) { //if update is true and patches is false, it will still copy the unchanged data over
			if(patches) {
				memcpy(pbuf, cpup, size);
				void *fp;
				for(GpuClonePatch_t *p = &patches[0]; p->offset>=0; p++) {
					memcpy((void*)&fp, (char*)cpup + p->offset, sizeof(void *));
					void *dpp = GpuFindCloneDevice(fp, dev);
					if (!dpp)
						GpuErr("Missing field clone when patching");
					memcpy(pbuf + p->offset, (void *)&dpp, sizeof(void *));
				}
				GpuMemcpyHostToDevice(dclones[dev], pbuf, size);
			} else {
				GpuMemcpyHostToDevice(dclones[dev], cpup, size);
			}
		}
	}
	for(int tid = 0; tid < GpuNumCpuThreads; tid++) {
		crp->info[tid].gpup = dclones[GpuThreadDevice[tid]];
		crp->info[tid].data = 0;
	}
	delete[] dclones;
	GpuSetDevice(savedev);
	if(pbuf) delete[] pbuf;
}

//
// Remove all clones associated with cpup
//
void GpuFreeClones(void *cpup) {
	GpuCloneRecord_t *crp = getclonerecord(cpup);
	GpuCloneInfo_t *info = crp->info;
	if(!crp) return; // nothing to do

	for(int tid = 0; tid < GpuNumCpuThreads; tid++) {
		void *gpup = info[tid].gpup;
		if(!gpup) continue;
		GpuFreeDevice(gpup);
		info[tid].gpup = nullptr;
		// Now make sure we get the other thread slots that have the same clone
		int dev = GpuThreadDevice[tid];
		for(int xtid = tid + 1; xtid < GpuNumCpuThreads; xtid++) {
			if(GpuThreadDevice[xtid] == dev && gpup == info[xtid].gpup) {
				info[xtid].gpup = nullptr; // prevent double free
			}
		}
	}
}

#if 0
void GpuOmpDelegate(size_t cnt, int &startb, int &bcnt, size_t &tstart, size_t &tcnt) {
	int tid = omp_get_thread_num;
	totalblocks = GpuThreads2BlockCount(cnt);
	// Figure out which blocks this this OMP thread services
	tblocks = totalblocks / GpuNumCpuThreads;
	leftover = tblocks - tblocks * GpuNumCpuThreads;
	startb = std::min(tid, leftover) + tblocks * tid;
}

#endif
