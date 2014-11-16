#include "gpu-pt-kernels.h"

#include "util.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <librta/intersect.h>
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
	
	__global__ void combine_color_samples(float3 *data, uint w, uint h, float3 *sample, int samples_already_accumulated) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		float3 sofar = data[id];
		data[id] = (samples_already_accumulated * sofar + sample[id]) / (samples_already_accumulated + 1);
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

void combine_color_samples(float3 *accum, uint w, uint h, float3 *sample, int samples_already_accumulated) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::combine_color_samples<<<blocks, threads>>>(accum, w, h, sample, samples_already_accumulated);
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
												halton_pool2f uniform_random, int sample, float3 *throughput, float3 *ray_diff_org, float3 *ray_diff_dir) {
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
			float3 dir;
			bool reflection = false;
			if (reflection == true) {
				org_dir = transform_to_tangent_frame(org_dir, T, B, N);
				dir = org_dir;
				dir.z = -dir.z;
				dir = transform_from_tangent_frame(dir, T, B, N);
			}
			else {
				float3 refl = reflect(org_dir, N);
				make_tangent_frame(refl, T, B);
				float2 rnd = uniform_random.data[(7*id + sample) % uniform_random.N];
				float n = 16;
				dir.z = 1.0f-powf(rnd.x, n+1.0f);
				dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*rnd.y);
				dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*rnd.y);
				dir = transform_from_tangent_frame(dir, T, B, refl);
// 				if (gid.x > 600 && gid.x < 610 && gid.y > 350 && gid.y < 360)
// 					printf("(%04d %04d) %6.6f %6.6f\n"
// 					       "            %6.6f %6.6f %6.6f\n"
// 					       "            %6.6f %6.6f %6.6f\n",
// 						   gid.x, gid.y,
// 						   rnd.x, rnd.y,
// 						   N.x, N.y, N.z,
// 						   dir.x, dir.y, dir.z);
			}
			float len = length_of_vector(dir);
			dir /= len;
			
			P += 0.01*dir;
			ray_orig[id] = P;
			ray_dir[id]  = dir;
			max_t[id]    = FLT_MAX;

			// upper ray
			float3 other_org = ray_diff_org[id];
			float3 other_dir = ray_diff_dir[id];
			triangle_intersection<rta::cuda::simple_triangle> other_is;
			intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
			float3 other_P, other_bc;
			other_is.barycentric_coord(&other_bc);
			barycentric_interpolation(&other_P, &other_bc, &tri.a, &tri.b, &tri.c);
			ray_diff_org[id] = other_org;
			other_dir = reflect(other_dir, N);
			ray_diff_dir[id] = other_dir;
			
			// right ray
			other_org = ray_diff_org[w*h+id];
			other_dir = ray_diff_dir[w*h+id];
			intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
			other_is.barycentric_coord(&other_bc);
			barycentric_interpolation(&other_P, &other_bc, &tri.a, &tri.b, &tri.c);
			ray_diff_org[w*h+id] = other_org;
			other_dir = reflect(other_dir, N);
			ray_diff_dir[w*h+id] = other_dir;
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
								 halton_pool2f uniform_random, int sample, float3 *throughput, float *ray_diff_orig, float *ray_diff_dir) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::generate_random_path_sample<<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
														ti, triangles, uniform_random, sample, throughput,
														(float3*)ray_diff_orig, (float3*)ray_diff_dir);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}
