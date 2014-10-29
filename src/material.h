#ifndef __GI_GPU_MATERIAL_H__ 
#define __GI_GPU_MATERIAL_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace rta {
	namespace cuda {

		struct texture_data {
			unsigned int w, h;
			unsigned char *rgba;
			texture_data(int w, int h) : w(w), h(h), rgba(0) {
				checked_cuda(cudaMalloc(&rgba, 4*w*h));
			}
			void upload(unsigned char *src) {
				checked_cuda(cudaMemcpy(rgba, src, 4*w*h, cudaMemcpyHostToDevice));
			}
			__device__ float3 sample(float s, float t) {
				float x = s*w;
				float y = (1.0f-t)*h;
				int nearest_x = int(x);
				int nearest_y = int(y);
				nearest_x = nearest_x % w;
				nearest_y = nearest_y % h;
// 				while (nearest_x >= w) nearest_x -= w;
// 				while (nearest_y >= h) nearest_y -= h;
// 				while (nearest_x < 0) nearest_x += w;
// 				while (nearest_y < 0) nearest_y += h;
				return make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f,
								   float(rgba[4*(nearest_y*w+nearest_x)+1])/255.0f,
								   float(rgba[4*(nearest_y*w+nearest_x)+2])/255.0f);
// 				return make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f, 0, 0);
			}
		};

		struct material_t {
			float3 diffuse_color;
			texture_data *diffuse_texture;

			material_t() : diffuse_color(make_float3(0,0,0)), diffuse_texture(0) {
			}
		};

		//! convert standard rta cpu tracer materials to a version suitable for gpu use.
		material_t* convert_and_upload_materials();
		
		
		//! 
		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst);
	}
}

#endif

