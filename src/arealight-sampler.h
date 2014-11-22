#ifndef __AREALIGHT_SAMPLER_H__ 
#define __AREALIGHT_SAMPLER_H__ 

#include "util.h"

#include <libcgls/scene.h>

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace rta {
	namespace cuda {
		namespace cgls {

	
			struct rect_light {
				float3 center, dir, up, col;
				float2 wh;
			};

			rect_light* convert_and_upload_rectangular_area_lights(scene_ref scene, int &N);
			void update_rectangular_area_lights(scene_ref scene, rect_light *data, int N);

			void generate_rectlight_sample(int w, int h, rect_light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::halton_pool2f uniform01, float3 *potential_sample_contribution, int sample, int max_samples);
			void generate_rectlight_sample(int w, int h, rect_light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::lcg_random_state uniform01, float3 *potential_sample_contribution, int sample, int max_samples);
			void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
										float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, int sample);
			void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
										float3 *potential_sample_contribution, float3 *material_col, float3 *throughput, float3 *col_accum, int sample);
			void init_cuda_image_transfer(texture_ref tex);
			void copy_cuda_image_to_texture(int w, int h, float3 *col, float scale);
		}
	}
}

#endif

