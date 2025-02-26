cmake_minimum_required(VERSION 3.29)
project(placudareal CUDA)

# Set the CUDA toolkit root directory if not set
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/") # Update this path to your CUDA installation path
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(OPT "-O0")

# NVCC and CXX flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OPT} -Xcompiler /openmp:llvm -G -m64 -std=c++17 -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT} /openmp:llvm")
add_compile_definitions(_USE_MATH_DEFINES)

set(CUDA_SOURCES
        gpuutil.cu
        oneDArray.cu
        pla.cu
        miscfunctions.cu
        main.cu
        # Add other CUDA source files here
)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(placudareal ${CUDA_SOURCES})
target_link_libraries(placudareal ${CUDA_LIBRARIES})

set_target_properties(placudareal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)