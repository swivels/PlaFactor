cmake_minimum_required(VERSION 3.29)
project(placudareal CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(OPT "-O0")

# NVCC and CXX flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OPT} -Xcompiler /openmp:llvm -G -m64 -std=c++17 -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT} /openmp:llvm")
add_compile_definitions(_USE_MATH_DEFINES)

add_executable(placudareal main.cu pla.cu oneDArray.cu gpuutil.cu miscfunctions.cu)

#set_target_properties(CudaCodeKen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)