#include "material.h"

#include <stdlib.h>

using namespace std;

namespace rta {
	namespace cuda {

		namespace k {
			__global__ void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst) {
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
						float3 tex = mat.diffuse_texture->sample(T.x, T.y);
						out.x *= tex.x;
						out.y *= tex.y;
						out.z *= tex.z;
					}
				}
				dst[gid.y*w+gid.x] = out;
			}
		}

		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
			k::evaluate_material<<<blocks, threads>>>(w, h, ti, triangles, mats, dst);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
		}

	}
}
