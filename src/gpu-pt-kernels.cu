#include "gpu-pt-kernels.h"

#include "util.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <libhyb/trav-util.h>

using namespace rta;
using namespace rta::cuda;
using namespace gi;
using namespace gi::cuda;

namespace k {
	__global__ void reset_data(float3 *data, uint w, uint h, float3 val) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		data[id] = val;
	}
}

void reset_gpu_buffer(float3 *data, uint w, uint h, float3 val) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::reset_data<<<blocks, threads>>>(data, w, h, val);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}


namespace k {
	__global__ void generate_random_path_sample(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t,
												triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles,
												halton_pool2f uniform_random, int sample, float3 *throughput) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
		if (is.valid()) {
			float3 bc; 
			float3 P, N, T, B;
			rta::cuda::simple_triangle tri = triangles[is.ref];
			is.barycentric_coord(&bc);
			barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
			barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);

			make_tangent_frame(N, T, B);
			
			float3 org_dir = ray_dir[id];
			org_dir = transform_to_tangent_frame(org_dir, T, B, N);
// 			float3 dir = reflect(org_dir, make_float3(0,0,1));
			float3 dir = org_dir;
			dir.z = -dir.z;
			dir = transform_from_tangent_frame(dir, T, B, N);
			float len = length_of_vector(dir);
			dir /= len;
			if (gid.x > 500 && gid.x < 510 && gid.y > 400 && gid.y < 410)
			printf("- org: %6.6f %6.6f %6.6f\n  nor: %6.6f %6.6f %6.6f\n  ref: %6.6f %6.6f %6.6f\n",
				   org_dir.x, org_dir.y, org_dir.z,
				   N.x, N.y, N.z,
				   dir.x, dir.y, dir.z);
			P += 0.01*dir;
			ray_orig[id] = P;
			ray_dir[id]  = dir;
			max_t[id]    = FLT_MAX;
		}
		else {
			ray_dir[id]  = make_float3(0,0,0);
			ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			max_t[id] = -1;
			throughput[id] = make_float3(0,0,0);
		}

	}
}

void generate_random_path_sample(int w, int h, float *ray_orig, float *ray_dir, float *max_t,
								 triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles,
								 halton_pool2f uniform_random, int sample, float3 *throughput) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::generate_random_path_sample<<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
														ti, triangles, uniform_random, sample, throughput);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}
