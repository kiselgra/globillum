#include "material.h"

#include <stdlib.h>

using namespace std;

namespace rta {
	namespace cuda {

		namespace k {

			__device__ inline float3 operator-(const float3 &a) {
				return make_float3(-a.x, -a.y, -a.z);
			}

			__device__ inline float operator|(const float3 &a, const float3 &b) {
				return a.x*b.x + a.y*b.y + a.z*b.z;
			}

			__global__ void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float3 *ray_dir) {
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
						d *= (N | -dir);
						if (dx < dy) d = dy;
						float mm = floor(log2f(d));
						if (mm < 1) out.x = 1, out.y = out.z = 0;
						else if (mm < 2) out.y = 1, out.x = out.z = 0;
						else if (mm < 3) out.z = 1, out.x = out.y = 0;
						else if (mm < 4) out.x = out.y = 1, out.z = 0;
						else if (mm < 5) out.x = out.z = 1, out.z = 0;
						else if (mm < 6) out.y = out.z = 1, out.z = 0;
						else out.y = out.z = out.z = 1;
						/*
						float3 tex = mat.diffuse_texture->sample(T.x, T.y);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
						*/
					}
				}
				dst[gid.y*w+gid.x] = out;
			}
		}

		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, float2 rd_xy, float *ray_dir) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
			k::evaluate_material<<<blocks, threads>>>(w, h, ti, triangles, mats, dst, rd_xy, (float3*)ray_dir);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
		}

	}
}
