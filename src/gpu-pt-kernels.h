#ifndef __GI_GPU_PT_KERNELS_H__ 
#define __GI_GPU_PT_KERNELS_H__ 

#include "util.h"
#include "material.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

void reset_gpu_buffer(float3 *data, uint w, uint h, float3 val);
void combine_color_samples(float3 *accum, uint w, uint h, float3 *sample, int samples_already_accumulated);


void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, gi::cuda::halton_pool2f uniform_random, float3 *throughput, float3 *col_accum,
										  float *to_light, rta::triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::cuda::random_sampler_path_info pi);
void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, gi::cuda::lcg_random_state uniform_random, float3 *throughput, float3 *col_accum,
										  float *to_light, rta::triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::cuda::random_sampler_path_info pi);
void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, gi::cuda::mt_pool3f uniform_random, float3 *throughput, 
										  float3 *col_accum, float *to_light, rta::triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::cuda::random_sampler_path_info pi);

#endif

