#include "gpu-pt-kernels.h"

#include "util.h"
#include "material.h"

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
												rta::cuda::material_t *mats,
												halton_pool2f uniform_random, int sample, float3 *throughput, float3 *ray_diff_org, float3 *ray_diff_dir) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
		if (is.valid()) {
			float3 bc; 
			float3 P, N, T, B;
			float2 TC;
			rta::cuda::simple_triangle tri = triangles[is.ref];
			is.barycentric_coord(&bc);
			barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
			barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
			barycentric_interpolation(&TC, &bc, &tri.ta, &tri.tb, &tri.tc);
			normalize_vec3f(&N);

			make_tangent_frame(N, T, B);
			normalize_vec3f(&T);
			normalize_vec3f(&B);
			
			// eval ray differentials (stored below)
			float3 upper_org = ray_diff_org[id],
				   upper_dir = ray_diff_dir[id],
				   right_org = ray_diff_org[w*h+id],
				   right_dir = ray_diff_dir[w*h+id];
			triangle_intersection<rta::cuda::simple_triangle> other_is;
			float3 upper_P, right_P, other_bc;
			float2 upper_T, right_T;
			// - upper ray
			intersect_tri_opt_nocheck(tri, (vec3f*)&upper_org, (vec3f*)&upper_dir, other_is);
			other_is.barycentric_coord(&other_bc);
			barycentric_interpolation(&upper_T, &other_bc, &tri.ta, &tri.tb, &tri.tc);
			barycentric_interpolation(&upper_P, &other_bc, &tri.a, &tri.b, &tri.c);
			// - right ray
			intersect_tri_opt_nocheck(tri, (vec3f*)&right_org, (vec3f*)&right_dir, other_is);
			other_is.barycentric_coord(&other_bc);
			barycentric_interpolation(&right_T, &other_bc, &tri.ta, &tri.tb, &tri.tc);
			barycentric_interpolation(&right_P, &other_bc, &tri.a, &tri.b, &tri.c);
			// - store origins
			ray_diff_org[id] = upper_P;
			ray_diff_org[w*h+id] = right_P;
			
			// material components
			material_t mat = mats[tri.material_index];
			float3 diffuse = mat.diffuse_color,
				   specular = mat.specular_color;
			if (mat.diffuse_texture || mat.specular_texture) {
				float diff_x = fabsf(TC.x - upper_T.x);
				float diff_y = fabsf(TC.y - upper_T.y);
				diff_x = fmaxf(fabsf(TC.x - right_T.x), diff_x);
				diff_y = fmaxf(fabsf(TC.y - right_T.y), diff_y);
				float diff = fmaxf(diff_x, diff_y);
				if (mat.diffuse_texture)
					diffuse *= mat.diffuse_texture->sample_bilin_lod(TC.x, TC.y, diff, gid, blockIdx, threadIdx);
				if (mat.specular_texture)
					specular *= mat.specular_texture->sample_bilin_lod(TC.x, TC.y, diff, gid, blockIdx, threadIdx);
			}
			float pd = diffuse.x+diffuse.y+diffuse.z;
			float ps = specular.x+specular.y+specular.z;
			if (pd + ps > 1) {
				pd /= pd+ps;
				ps /= pd+ps;
			}
			// normalized to 1 (inkl absorption)
			float2 rnd = uniform_random.data[(3*id + sample) % uniform_random.N];
			bool diffuse_bounce = false,
				 specular_bounce = false;
			float P_component;
			if (rnd.y <= pd) {
				diffuse_bounce = true;
				P_component = pd;
			}
			else if (rnd.y <= pd+ps) {
				specular_bounce = true;
				P_component = ps;
			}
// 			diffuse_bounce = false;
// 			specular_bounce = true;
// 			P_component = 1;
		
			float3 org_dir = ray_dir[id];
			float3 dir;
			float n;
			float omega_z;
			rnd = uniform_random.data[(7*id + sample) % uniform_random.N];

			bool reflection = false;
			if (reflection) {
				org_dir = transform_to_tangent_frame(org_dir, T, B, N);
				dir = org_dir;
				dir.z = -dir.z;
				dir = transform_from_tangent_frame(dir, T, B, N);
				// store reflection ray differentials. (todo: change to broadened cone)
				ray_diff_dir[id] = reflect(upper_dir, N);
				ray_diff_dir[w*h+id] = reflect(right_dir, N);
			}
			else if (diffuse_bounce) {
				n = 1.0f;
				dir.z = powf(rnd.x, 1.0f/(n+1.0f));
				dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*rnd.y);
				dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*rnd.y);
				omega_z = dir.z;
				dir = transform_from_tangent_frame(dir, T, B, N);
				// store reflection ray differentials. (todo: change to broadened cone)
				ray_diff_dir[id] = reflect(upper_dir, N);
				ray_diff_dir[w*h+id] = reflect(right_dir, N);
			}
			else if (specular_bounce) {
				float3 refl = reflect(org_dir, N);
				make_tangent_frame(refl, T, B);
				n = 140.0f;
				dir.z = powf(rnd.x, 1.0f/(n+1.0f));
				dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*rnd.y);
				dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*rnd.y);
				omega_z = dir.z;
				dir = transform_from_tangent_frame(dir, T, B, refl);
				// store reflection ray differentials.
				ray_diff_dir[id] = reflect(upper_dir, N);
				ray_diff_dir[w*h+id] = reflect(right_dir, N);
				if ((dir|N)<0) specular_bounce = false;
			}
			if (diffuse_bounce||specular_bounce) {
				float len = length_of_vector(dir);
				dir /= len;

				P += 0.01*dir;
				ray_orig[id] = P;
				ray_dir[id]  = dir;
				max_t[id]    = FLT_MAX;
				throughput[id] *= (1.0f/P_component) * ((2.0f*float(M_PI))/((n+1.0f)*pow(omega_z,n)));

				return;
			}

// 			other_dir = 
// 			ray_diff_dir[id] = other_dir;
// 			other_dir = reflect(other_dir, N);
		}
		else {
		// fall through
		ray_dir[id]  = make_float3(0,0,0);
		ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		max_t[id] = -1;
		throughput[id] = make_float3(0,0,0);
		}

	}
}

void generate_random_path_sample(int w, int h, float *ray_orig, float *ray_dir, float *max_t,
								 triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles,
								 rta::cuda::material_t *mats,
								 halton_pool2f uniform_random, int sample, float3 *throughput, float *ray_diff_orig, float *ray_diff_dir) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::generate_random_path_sample<<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
														ti, triangles, mats, uniform_random, sample, throughput,
														(float3*)ray_diff_orig, (float3*)ray_diff_dir);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}
