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
	__device__ bool operator!=(const float3 &a, const float3 &b) {
		if (   fabsf(a.x - b.x) > 0.001 
			|| fabsf(a.y - b.y) > 0.001
			|| fabsf(a.z - b.z) > 0.001) return true;
		return false;
	}

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
			normalize_vec3f(&N);

			make_tangent_frame(N, T, B);
			normalize_vec3f(&T);
			normalize_vec3f(&B);
		
			float3 org_dir = ray_dir[id];
// 			float3 org_dir_2 = transform_to_tangent_frame(org_dir, T, B, N);
// 			float3 org_dir_3 = transform_from_tangent_frame(org_dir_2, T, B, N);

// 			if (org_dir != org_dir_3 ){//&& gid.x > 500 && gid.x < 510) {
// 				printf("(%d %d) --------------------\n"
// 					   "T    = [ %6.6f ; %6.6f ; %6.6f ]\n"
// 					   "B    = [ %6.6f ; %6.6f ; %6.6f ]\n"
// 					   "N    = [ %6.6f ; %6.6f ; %6.6f ]\n"
// 					   "d    = [ %6.6f ; %6.6f ; %6.6f ]\n"
// 					   "d_t  = [ %6.6f ; %6.6f ; %6.6f ]\n"
// 					   "d_T  = [ %6.6f ; %6.6f ; %6.6f ]\n",
// 					   gid.x, gid.y, T.x, T.y, T.z, B.x, B.y, B.z, N.x, N.y, N.z,
// 					   org_dir.x, org_dir.y, org_dir.z,
// 					   org_dir_2.x, org_dir_2.y, org_dir_2.z,
// 					   org_dir_3.x, org_dir_3.y, org_dir_3.z);
// 			}

			float3 org_dir_ts = transform_to_tangent_frame(org_dir, T, B, N);
			float3 refl_ts = reflect(org_dir_ts, make_float3(0,0,1));
			float3 refl1 = transform_from_tangent_frame(refl_ts, T, B, N);
			
			float3 refl2 = reflect(org_dir, N);
			if (refl1 != refl2&& gid.x > 500 && gid.x < 510) {
				printf("(%d %d) --------------------\n"
					   "T    = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "B    = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "N    = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "d    = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "d_t  = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "r_t  = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "r_1  = [ %6.6f ; %6.6f ; %6.6f ]\n"
					   "r_2  = [ %6.6f ; %6.6f ; %6.6f ]\n",
					   gid.x, gid.y, T.x, T.y, T.z, B.x, B.y, B.z, N.x, N.y, N.z,
					   org_dir.x, org_dir.y, org_dir.z,
					   org_dir_ts.x, org_dir_ts.y, org_dir_ts.z,
					   refl_ts.x, refl_ts.y, refl_ts.z,
					   refl1.x, refl1.y, refl1.z,
					   refl2.x, refl2.y, refl2.z);
			}
			float3 dir = refl2;
// 			float3 dir = reflect(org_dir, make_float3(0,0,1));
// 			float3 dir = org_dir;
// 			dir.z = -dir.z;
// 			dir = transform_from_tangent_frame(dir, T, B, N);
			float len = length_of_vector(dir);
			dir /= len;
// 			if (gid.x > 500 && gid.x < 510 && gid.y > 400 && gid.y < 410)
// 			printf("- org: %6.6f %6.6f %6.6f\n  nor: %6.6f %6.6f %6.6f\n  ref: %6.6f %6.6f %6.6f\n",
// 				   org_dir.x, org_dir.y, org_dir.z,
// 				   N.x, N.y, N.z,
// 				   dir.x, dir.y, dir.z);
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
