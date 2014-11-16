#ifndef __GI_GPU_PT_KERNELS_H__ 
#define __GI_GPU_PT_KERNELS_H__ 

#include "util.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

void reset_gpu_buffer(float3 *data, uint w, uint h, float3 val);

void generate_random_path_sample(int w, int h, float *ray_orig, float *ray_dir, float *max_t,
								 rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles,
								 gi::cuda::halton_pool2f uniform_random, int sample, float3 *throughput, float *ray_diff_orig, float *ray_diff_dir);

#endif

