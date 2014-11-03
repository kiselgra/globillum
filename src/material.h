#ifndef __GI_GPU_MATERIAL_H__ 
#define __GI_GPU_MATERIAL_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
			__device__ float3 sample_bilin_lod(float s, float t, float diff, int2 gid, uint3 bid, uint3 tid) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
				  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher    
#endif
				int W = w;
			   	int H = h;
				float lod = log2f(diff * fmaxf(w,h));
				lod = fminf(lod, float(max_mm));
// 				if (lod < 1) return make_float3(diff,0,0);
// 				else if (lod < 2) return make_float3(0,1,0);
// 				else if (lod < 2) return make_float3(0,0,1);
// 				else if (lod < 2) return make_float3(1,0,1);
// 				else if (lod < 2) return make_float3(1,1,0);
// 				return make_float3(1,1,1);

// 				int WW = w;
// 			   	int HH = h;
// 				    printf("Hello thread %d, f=%f\n", gid.x, 0.1f);
// 				if (gid.x > 100 && gid.x < 110 && gid.y > 100 && gid.y < 110)
// 					printf("foo\n");
// 					printf("WW=%d, W=%u, w=%u, HH=%d, H=%u, h=%u\n", WW, W, w, HH, H, h);
// 				if (gid.x > 100 && gid.x < 110 && gid.y > 100 && gid.y < 110)
// 					printf("WW=%d, w=%u, HH=%d, h=%u\n", W, w, H, h);
				unsigned int offset = 0;
				for (int i = 0; i < lod; ++i) {
					offset += W*H;
					W = (W+1)/2;
					H = (H+1)/2;
				}
				if (s < 0)  s += -truncf(s)+1;
				if (s >= 1) s -= truncf(s);
				if (t < 0)  t += -truncf(t)+1;
				if (t >= 1) t -= truncf(t);
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
				// nearest is in bounds.
				if (other_x > W) other_x -= W;
				if (other_x < 0) other_x += W;
				if (other_y > H) other_y -= H;
				if (other_y < 0) other_y += H;
// 				if (bid.x == 25 && bid.y == 0 && tid.y==4) {
// 					587,341
// 					gid=(587,341)   0.226815 0.859818 -8983.741211 880.453125
// 					gid=(587,341)   n_x=-8983:-8983 W=1024 n_y=880:880
// 					gid=(587,341)   n_x = -8983,    n_y = 880
//
// 					gid=(587,341)   -16.773186 0.859818 -8983.741211 880.453125
// 					gid=(587,341)   n_y=-8983:-8983 W=1024 n_x=880:880
// 					gid=(587,341)   n_x = -8983,    n_y = 880
//
// 					gid=(587,341)   -8.773185 0.859818 -8983.741211 880.453125
// 					gid=(587,341)   n_y=-8983:-8983 W=1024 n_x=880:880
// 					gid=(587,341)   n_x = -8983,    n_y = 880
//
// 				if (gid.x == 587 && gid.y == 341) {
// 					printf("gid=(%d,%d)\t%6.6f %6.6f %6.6f %6.6f\n", gid.x, gid.y, s, 1.0f-t, x, y);
// 					printf("gid=(%d,%d)\tn_x=%d:%d W=%d n_y=%d:%d\n", gid.x, gid.y, nearest_x, int(x), W, nearest_y, int(y));
// 					printf("gid=(%d,%d)\tn_x = %d,\tn_y = %d\n", gid.x, gid.y, nearest_x, nearest_y);
// 				}
				if (nearest_y >= H || nearest_x >= W || nearest_y < 0 || nearest_x < 0)
					printf("gid=(%d,%d)\tn_x = %d,\tn_y = %d\n", gid.x, gid.y, nearest_x, nearest_y);
				float3 a00 = make_float3(float(rgba[4*(offset + nearest_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+nearest_x)+2])/255.0f);
				float3 a01 = make_float3(float(rgba[4*(offset + nearest_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+other_x)+2])/255.0f);
				float3 a10 = make_float3(float(rgba[4*(offset + other_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset + other_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset + other_y*W+nearest_x)+2])/255.0f);
				float3 a11 = make_float3(float(rgba[4*(offset + other_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset + other_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset + other_y*W+other_x)+2])/255.0f);
// 				float3 a00, a01, a10, a11;
// 				a00 = a01 = a10 = a11 = make_float3(0,0,0);
				float3 a0x = wx*a00 + (1.0f-wx)*a01;
				float3 a1x = wx*a10 + (1.0f-wx)*a11;
				return wy*a0x + (1.0f-wy)*a1x;
			}
			/*
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
				return a00;//wy*a0x + (1.0f-wy)*a1x;
			}
			*/
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
		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float *ray_org, float *ray_dirs);
	}
}

#endif

