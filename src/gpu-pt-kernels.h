#ifndef __GI_GPU_PT_KERNELS_H__ 
#define __GI_GPU_PT_KERNELS_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

void reset_gpu_buffer(float3 *data, uint w, uint h, float3 val);

#endif

