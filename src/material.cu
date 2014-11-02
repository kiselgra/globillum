#include "material.h"

#include <stdlib.h>

using namespace std;

namespace rta {
	namespace cuda {
			
		// 
		// material evaluation
		// 

		namespace k {

			__device__ inline float3 operator-(const float3 &a) {
				return make_float3(-a.x, -a.y, -a.z);
			}

			__device__ inline float operator|(const float3 &a, const float3 &b) {
				return a.x*b.x + a.y*b.y + a.z*b.z;
			}

			__global__ void evaluate_material_bilin(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float3 *ray_dir) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
				float3 out = make_float3(0,0,0);
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

			__global__ void evaluate_material_trilin(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float3 *ray_dir) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
				float3 out = make_float3(0,0,0);
				if (is.valid()) {
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
						// ray diffs
						float dx = is.t * rd_xy.x;
						float dy = is.t * rd_xy.y;
						float d = dx;
						float3 dir = ray_dir[gid.y*w+gid.x];
// 						d *= (N | -dir);
						if (dx < dy) d = dy;
						float mm = 0;//floor(log2f(d));
// 						if (mm < 1) out.x = 1, out.y = out.z = 0;
// 						else if (mm < 2) out.y = 1, out.x = out.z = 0;
// 						else if (mm < 3) out.z = 1, out.x = out.y = 0;
// 						else if (mm < 4) out.x = out.y = 1, out.z = 0;
// 						else if (mm < 5) out.x = out.z = 1, out.z = 0;
// 						else if (mm < 6) out.y = out.z = 1, out.z = 0;
// 						else out.y = out.z = out.z = 1;
						float3 tex = mat.diffuse_texture->sample_bilin_lod(T.x, T.y, (int)mm, gid, blockIdx, threadIdx);
// 						float3 tex = mat.diffuse_texture->sample_bilin(T.x, T.y);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
// 						if (tri.material_index < 9)
// 							out.x = tri.material_index/8.0f, out.y = out.z = 0;
// 						else if (tri.material_index < 18)
// 							out.y = (tri.material_index-9)/8.0f, out.x = out.z = 0;
// 						else 
// 							out.z = (tri.material_index-18)/8.0f, out.x = out.y = 0;
					}
				}
				dst[gid.y*w+gid.x] = out;
			}
		}

		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float *ray_dir) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
// 			k::evaluate_material_bilin<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, rd_xy, (float3*)ray_dir);
			k::evaluate_material_trilin<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, rd_xy, (float3*)ray_dir);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
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
				printf("mm: %d x %d \t --> %d x %d\n", w, h, target_w, target_h);
				printf("    %d,\t %d\n", s_offset, d_offset);
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
}
