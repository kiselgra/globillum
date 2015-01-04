#include "material.h"

#include "vars.h"

#include <librta/intersect.h>

#include <stdlib.h>

using namespace std;

namespace rta {
	namespace cuda {
			
		// 
		// material evaluation
		// 

		namespace k {

// 			__device__ inline float3 operator-(const float3 &a) {
// 				return make_float3(-a.x, -a.y, -a.z);
// 			}
// 
// 			__device__ inline float operator|(const float3 &a, const float3 &b) {
// 				return a.x*b.x + a.y*b.y + a.z*b.z;
// 			}

			__global__ void evaluate_material_bilin(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float3 background) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
				float3 out = background;
				if (is.valid()) {
					cuda::simple_triangle tri = triangles[is.ref];
					material_t mat = mats[tri.material_index];
					out = mat.diffuse_color;
					if (mat.diffuse_texture) {
						float3 bc; 
						is.barycentric_coord(&bc);
						const float2 &ta = tri.ta;
						const float2 &tb = tri.tb;
						const float2 &tc = tri.tc;
						float2 T;
						barycentric_interpolation(&T, &bc, &ta, &tb, &tc);
						float3 tex = mat.diffuse_texture->sample_bilin(T.x, T.y);
// 						float3 tex = mat.diffuse_texture->sample_nearest(T.x, T.y);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
					}
				}
				dst[gid.y*w+gid.x] = out;
			}

			__global__ void evaluate_material_bilin_lod(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, 
														float3 *dst, float3 *ray_org, float3 *ray_dir, float3 background) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
				float3 out = background;
				if (is.valid()) {
// 					printf("(%03d %03d) t %6.6f T %d\n", is.t, is.ref);
// 					return;
// 					out = tri.a;
					out = make_float3(is.ref, is.ref, is.ref);
					dst[gid.y*w+gid.y] = out;
					cuda::simple_triangle tri = triangles[is.ref];
					material_t mat = mats[tri.material_index];
					out = mat.diffuse_color;
					if (mat.diffuse_texture) {
						float3 bc; 
						is.barycentric_coord(&bc);
						// tex coord
						const float2 &ta = tri.ta;
						const float2 &tb = tri.tb;
						const float2 &tc = tri.tc;
						float2 T;
						barycentric_interpolation(&T, &bc, &ta, &tb, &tc);
						// normal
						const float3 &na = tri.na;
						const float3 &nb = tri.nb;
						const float3 &nc = tri.nc;
						float3 N;
						barycentric_interpolation(&N, &bc, &na, &nb, &nc);
						// eval other rays
						int ox = 1, oy = 1;
						if (gid.x == w-1) ox = -1;
						if (gid.y == h-1) oy = -1;
						// upper ray
						float3 other_org = ray_org[(gid.y+oy)*w+gid.x];
						float3 other_dir = ray_dir[(gid.y+oy)*w+gid.x];
						triangle_intersection<cuda::simple_triangle> other_is;
						intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
						float2 other_T;
						float3 other_bc;
						other_is.barycentric_coord(&other_bc);
						barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
						float diff_x = fabsf(T.x - other_T.x);
						float diff_y = fabsf(T.y - other_T.y);
						// right ray
						other_org = ray_org[(gid.y)*w+gid.x+ox];
						other_dir = ray_dir[(gid.y)*w+gid.x+ox];
						intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
						other_is.barycentric_coord(&other_bc);
						barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
						diff_x = fmaxf(fabsf(T.x - other_T.x), diff_x);
						diff_y = fmaxf(fabsf(T.y - other_T.y), diff_y);
						float diff = fmaxf(diff_x, diff_y);
						// access texture
						float3 tex = mat.diffuse_texture->sample_bilin_lod(T.x, T.y, diff);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
					}
				}
				dst[gid.y*w+gid.x] = out;
			}

			__host__ __device__ __forceinline__
			void pixel_evaluate_material_bilin_lod(int2 gid, 
												   int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
												   cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, 
												   float3 *ray_org, float3 *ray_dir, float3 *ray_diff_org, float3 *ray_diff_dir, 
												   float3 background) {
				triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
				float3 out = background;
				if (is.valid()) {
// 					if (is.ref > 331718) {
// 						printf("x (%03d %03d) t %6.6f T %d\n", gid.x, gid.y, is.t, is.ref);
// 						return;
// 					}
					cuda::simple_triangle tri = triangles[is.ref];
					material_t mat = mats[tri.material_index];
					out = mat.diffuse_color;
					if (mat.diffuse_texture) {
						float3 bc; 
						is.barycentric_coord(&bc);
						// tex coord
						const float2 &ta = tri.ta;
						const float2 &tb = tri.tb;
						const float2 &tc = tri.tc;
						float2 T;
						barycentric_interpolation(&T, &bc, &ta, &tb, &tc);
						// normal
						const float3 &na = tri.na;
						const float3 &nb = tri.nb;
						const float3 &nc = tri.nc;
						float3 N;
						barycentric_interpolation(&N, &bc, &na, &nb, &nc);
						// eval other rays
						// - upper ray
						float3 other_org = ray_diff_org[gid.y*w+gid.x];
						float3 other_dir = ray_diff_dir[gid.y*w+gid.x];
						triangle_intersection<cuda::simple_triangle> other_is;
						intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
						float2 other_T;
						float3 other_bc;
						other_is.barycentric_coord(&other_bc);
						barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
						float diff_x = fabsf(T.x - other_T.x);
						float diff_y = fabsf(T.y - other_T.y);
						// - right ray
						other_org = ray_diff_org[w*h+gid.y*w+gid.x];
						other_dir = ray_diff_dir[w*h+gid.y*w+gid.x];
						intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
						other_is.barycentric_coord(&other_bc);
						barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
						diff_x = fmaxf(fabsf(T.x - other_T.x), diff_x);
						diff_y = fmaxf(fabsf(T.y - other_T.y), diff_y);
						float diff = fmaxf(diff_x, diff_y);
// 						diff = 0;
// access texture
// 						float3 tex = mat.specular_texture->sample_bilin_lod(T.x, T.y, diff, gid, blockIdx, threadIdx);
						float3 tex = mat.diffuse_texture->sample_bilin_lod(T.x, T.y, diff);
// 						printf("mat %03d %d %d\n", tri.material_index, mat.diffuse_texture->w, mat.diffuse_texture->h);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
// 						out.x = T.x;
// 						out.y = T.y;
// 						out.z = 0;
					}
				}
				dst[gid.y*w+gid.x] = out;
			}

			__global__ void evaluate_material_bilin_lod(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, 
														float3 *dst, float3 *ray_org, float3 *ray_dir, float3 *ray_diff_org, float3 *ray_diff_dir, float3 background) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				pixel_evaluate_material_bilin_lod(gid, 
												  w, h, ti, triangles, mats, dst, ray_org, ray_dir, ray_diff_org, ray_diff_dir, 
												  background);
			}
		}

		void evaluate_material_bilin(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float3 background) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
			k::evaluate_material_bilin<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, background);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
		}

		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float *ray_org, float *ray_dir, float3 background) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
// 			k::evaluate_material_bilin<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, (float3*)ray_dir);
			k::evaluate_material_bilin_lod<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, (float3*)ray_org, (float3*)ray_dir, background);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
		}

		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float *ray_org, float *ray_dir, float *ray_diff_org, float *ray_diff_dir, float3 background) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
			k::evaluate_material_bilin_lod<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, (float3*)ray_org, (float3*)ray_dir, (float3*)ray_diff_org, (float3*)ray_diff_dir, background);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
			exit(1);
		}

		
		// 
		// mip mapping
		// 

		namespace k {
			__global__ void mipmap(int dst_w, int dst_h, int d_offset, int s_offset, int src_w, int src_h, uchar4 *data) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= dst_w || gid.y >= dst_h) return;
				int X = 2*gid.x,
					Y = 2*gid.y;
				uchar4 a00 = data[s_offset + Y*src_w + X],
					   a01, a10, a11;
				int    w00 = 1, w01 = 1, w10 = 1, w11 = 1;
				if (X+1 < src_w) a01 = data[s_offset + Y*src_w + X+1];
				else             w00 = 0;
				if (Y+1 < src_h) a10 = data[s_offset + (Y+1)*src_w + X];
				else             w10 = 0;
				if (X+1 < src_w && Y+1 < src_h) a11 = data[s_offset + (Y+1)*src_w + X+1];
				else                            w11 = 0;
				uint tmp, W = w00+w01+w10+w11;
				tmp = w00*uint(a00.x) + w01*uint(a01.x) + w10*uint(a10.x) + w11*uint(a11.x);   a00.x = tmp/W;
				tmp = w00*uint(a00.y) + w01*uint(a01.y) + w10*uint(a10.y) + w11*uint(a11.y);   a00.y = tmp/W;
				tmp = w00*uint(a00.z) + w01*uint(a01.z) + w10*uint(a10.z) + w11*uint(a11.z);   a00.z = tmp/W;
				tmp = w00*uint(a00.w) + w01*uint(a01.w) + w10*uint(a10.w) + w11*uint(a11.w);   a00.w = tmp/W;
				data[d_offset + gid.y*dst_w+gid.x] = a00;//make_uchar4(128, 64, 32, 255);
			}
		}

		void compute_mipmaps(texture_data *tex) {
			int w = tex->w, h = tex->h;
			int s_offset = 0;
			for (int i = 0; i < tex->max_mm; ++i) {
				int target_w = (w+1)/2,
					target_h = (h+1)/2;
				int d_offset = s_offset + w*h;
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(target_w, target_h, threads);
				k::mipmap<<<blocks, threads>>>(target_w, target_h, d_offset,
											   s_offset, w, h, (uchar4*)tex->rgba);
				w = target_w;
				h = target_h;
				s_offset = d_offset;
			}
		}

	}
		
	void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
						   cuda::material_t *mats, float3 *dst, float3 *ray_org, float3 *ray_dir, 
						   float3 *ray_diff_org, float3 *ray_diff_dir, float3 background) {
		#pragma omp parallel for
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				cuda::k::pixel_evaluate_material_bilin_lod(make_int2(x, y), 
														   w, h, ti, triangles, mats, dst, 
														   (float3*)ray_org, (float3*)ray_dir, 
														   (float3*)ray_diff_org, (float3*)ray_diff_dir, background);
			}
		}
	}

}
