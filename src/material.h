#ifndef __GI_GPU_MATERIAL_H__ 
#define __GI_GPU_MATERIAL_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace rta {
	namespace cuda {

		struct texture_data {
			unsigned int w, h;
			unsigned char *rgba;
			int max_mm;
			texture_data(int w, int h) : w(w), h(h), rgba(0) {
				checked_cuda(cudaMalloc(&rgba, 12*w*h));	// 6: 4 components times 1.5 for mip maps.
				int min = w<h?w:h;
				max_mm = int(floor(log2f(float(min))));
			}
			void upload(unsigned char *src) {
				checked_cuda(cudaMemcpy(rgba, src, 4*w*h, cudaMemcpyHostToDevice));
			}
			__device__ float3 sample_nearest(float s, float t) {
				float x = s*w;
				float y = (1.0f-t)*h;
				int nearest_x = int(x);
				int nearest_y = int(y);
				nearest_x = nearest_x % w;
				nearest_y = nearest_y % h;
				float3 a00 = make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+2])/255.0f);
				return a00;
			}
			__device__ float3 sample_bilin(float s, float t) {
				float x = s*w;
				float y = (1.0f-t)*h;
				int nearest_x = int(x);
				int nearest_y = int(y);
				int other_x = nearest_x+1;
				int other_y = nearest_y+1;
				if (x-floorf(x) < .5) other_x = nearest_x-1;
				if (y-floorf(y) < .5) other_y = nearest_y-1;
				// wx = 1.0 at floorf(x)=.5f.
				// wx = 0.5 at floorf(x)=.0f.
				float wx = fabsf(float(other_x)+.5 - x);
				float wy = fabsf(float(other_y)+.5 - y);
				nearest_x = nearest_x % w;
				nearest_y = nearest_y % h;
				other_x = other_x % w;
				other_y = other_y % h;
				float3 a00 = make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+2])/255.0f);
				float3 a01 = make_float3(float(rgba[4*(nearest_y*w+other_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+other_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+other_x)+2])/255.0f);
				float3 a10 = make_float3(float(rgba[4*(other_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(other_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(other_y*w+nearest_x)+2])/255.0f);
				float3 a11 = make_float3(float(rgba[4*(other_y*w+other_x)+0])/255.0f,
										 float(rgba[4*(other_y*w+other_x)+1])/255.0f,
										 float(rgba[4*(other_y*w+other_x)+2])/255.0f);
				float3 a0x = wx*a00 + (1.0f-wx)*a01;
				float3 a1x = wx*a10 + (1.0f-wx)*a11;
				return wy*a0x + (1.0f-wy)*a1x;
			}
			__device__ float3 sample_bilin_lod(float s, float t, int lod) {
				lod = 0;
				int W = w, H = h, offset = 0;
// 				for (int i = 0; i < lod; ++i) {
// 					offset += W*H;
// 					W = (W+1)/2;
// 					H = (H+1)/2;
// 				}

				float x = s*W;
				float y = (1.0f-t)*H;
				int nearest_x = int(x);
				int nearest_y = int(y);
				int other_x = nearest_x+1;
				int other_y = nearest_y+1;
				if (x-floorf(x) < .5) other_x = nearest_x-1;
				if (y-floorf(y) < .5) other_y = nearest_y-1;
				// wx = 1.0 at floorf(x)=.5f.
				// wx = 0.5 at floorf(x)=.0f.
				float wx = fabsf(float(other_x)+.5 - x);
				float wy = fabsf(float(other_y)+.5 - y);
				nearest_x = nearest_x % W;
				nearest_y = nearest_y % H;
				other_x = other_x % W;
				other_y = other_y % H;
				float3 a00 = make_float3(float(rgba[4*(offset+nearest_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset+nearest_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset+nearest_y*W+nearest_x)+2])/255.0f);
				float3 a01 = make_float3(float(rgba[4*(offset+nearest_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset+nearest_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset+nearest_y*W+other_x)+2])/255.0f);
				float3 a10 = make_float3(float(rgba[4*(offset+other_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset+other_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset+other_y*W+nearest_x)+2])/255.0f);
				float3 a11 = make_float3(float(rgba[4*(offset+other_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset+other_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset+other_y*W+other_x)+2])/255.0f);
				float3 a0x = wx*a00 + (1.0f-wx)*a01;
				float3 a1x = wx*a10 + (1.0f-wx)*a11;
				return wy*a0x + (1.0f-wy)*a1x;
			}
// 			__device__ float3 sample_bilin_lod(float s, float t, int lod) {
// 				float x = s*W;
// 				float y = (1.0f-t)*H;
// 				
// 				int nearest_x = int(x);
// 				int nearest_y = int(y);
// 				nearest_x = nearest_x % W;
// 				nearest_y = nearest_y % H;
// 				return make_float3(float(rgba[4*(offset + nearest_y*W+nearest_x)+0])/255.0f,
// 								   float(rgba[4*(offset + nearest_y*W+nearest_x)+1])/255.0f,
// 								   float(rgba[4*(offset + nearest_y*W+nearest_x)+2])/255.0f);
// 			}
		};
		void compute_mipmaps(texture_data *tex);

		struct material_t {
			float3 diffuse_color;
			texture_data *diffuse_texture;

			material_t() : diffuse_color(make_float3(0,0,0)), diffuse_texture(0) {
			}
		};

		//! convert standard rta cpu tracer materials to a version suitable for gpu use.
		material_t* convert_and_upload_materials();
		
		
		//! 
		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float *ray_dirs);
	}
}

#endif

