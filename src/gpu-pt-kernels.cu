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
	__device__ bool operator!=(const float3 &a, const float3 &b) {
		if (   fabsf(a.x - b.x) > 0.001 
			|| fabsf(a.y - b.y) > 0.001
			|| fabsf(a.z - b.z) > 0.001) return true;
		return false;
	}


	/*! \brief This kernel computes the current path's contribution (after the
	 *  shadowray has been cast) and sets up the next ray to create a longer
	 *  path.
	 *
	 *  The outline is as follows:
	 *  - As the surface color has to be computed for path-throughput as well
	 *    as for illumination we compute it first.
	 *  - Then we check if the light is visible from the original intersection
	 *    point.
	 *    - If it is, we compute the illumination and evaluate the complete brdf
	 *      (which is the phong brdf in our case).
	 *    - Furthermore, the light's contribution has to be weighted by the
	 *      path throughput and added to the current path's accumulation buffer.
	 *    - (MIS could be integrated here if we cast two shadow rays (including
	 *      further ray and intersection buffers).
	 *  - Next we compute the new bounce direction by sampling the phong brdf.
	 */
	template<typename rng_t> __global__ void 
	compute_path_contribution_and_bounce(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t, float3 *ray_diff_org, float3 *ray_diff_dir,
										 triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										 rta::cuda::material_t *mats, rng_t uniform_random, float3 *throughput, float3 *col_accum,
										 float3 *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										 float3 *potential_sample_contribution, random_sampler_path_info pi) {
		// general setup, early out if the path intersection is invalid.
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
		if (is.valid()) {

			// load hit triangle and compute hitpoint geometry
			rta::cuda::simple_triangle tri = triangles[is.ref];
			float2 TC;
			float3 bc, P, N;
			is.barycentric_coord(&bc);
			barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
			barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
			barycentric_interpolation(&TC, &bc, &tri.ta, &tri.tb, &tri.tc);
			normalize_vec3f(&N);
				
			// eval ray differentials (stored below)
			float3 upper_org = ray_diff_org[id],
				   upper_dir = ray_diff_dir[id],
				   right_org = ray_diff_org[w*h+id],
				   right_dir = ray_diff_dir[w*h+id];
			triangle_intersection<rta::cuda::simple_triangle> other_is;
			float3 upper_P, right_P;
			float2 upper_T, right_T;
			// - upper ray
			intersect_tri_opt_nocheck(tri, (vec3f*)&upper_org, (vec3f*)&upper_dir, other_is);
			other_is.barycentric_coord(&bc);
			barycentric_interpolation(&upper_T, &bc, &tri.ta, &tri.tb, &tri.tc);
			barycentric_interpolation(&upper_P, &bc, &tri.a, &tri.b, &tri.c);
			// - right ray
			intersect_tri_opt_nocheck(tri, (vec3f*)&right_org, (vec3f*)&right_dir, other_is);
			other_is.barycentric_coord(&bc);
			barycentric_interpolation(&right_T, &bc, &tri.ta, &tri.tb, &tri.tc);
			barycentric_interpolation(&right_P, &bc, &tri.a, &tri.b, &tri.c);
			// - store origins
			ray_diff_org[id] = upper_P;
			ray_diff_org[w*h+id] = right_P;
		
			// load and evaluate material
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
			float sum = diffuse.x+diffuse.y+diffuse.z+specular.x+specular.y+specular.z;;
			if (sum > 1.0f) {
				diffuse /= sum;
				specular /= sum;
			}

			// add lighting to accumulation buffer
			float3 org_dir = ray_dir[id];
			normalize_vec3f(&org_dir);
			float3 R = reflect(org_dir, N);
			// attention: recycling of 'is'
			is = shadow_ti[id];
			float3 TP = throughput[id];
			const float shininess = 100.0f;
			if (!is.valid()) {
				float3 prev = col_accum[id];
				float3 weight = potential_sample_contribution[id];
				// attention: we need the throughput *before* the bounce
				float3 curr = TP * weight;
				float3 light_dir = to_light[id];
				normalize_vec3f(&light_dir);
				// the whole geometric term is already computed in potential_sample_contribution.
				float3 brdf = diffuse * float(M_1_PI)
				            + (shininess + 1)*specular * 0.5 * M_1_PI * pow(fmaxf((R|light_dir), 0.0f), shininess);
				col_accum[id] = prev + brdf * curr;
			}

			// compute next path segment by sampling the brdf
			float3 random = next_random3f(uniform_random, id, pi);
			float pd = diffuse.x+diffuse.y+diffuse.z;
			float ps = specular.x+specular.y+specular.z;
			if (pd + ps > 1) {
				pd /= pd+ps;
				ps /= pd+ps;
			}
			bool diffuse_bounce = false,
				 specular_bounce = false;
			float P_component;
			if (random.z <= pd) {
				diffuse_bounce = true;
				P_component = pd;
			}
			else if (random.z <= pd+ps) {
				specular_bounce = true;
				P_component = ps;
			}

			float3 T, B;
			make_tangent_frame(N, T, B);
			normalize_vec3f(&T);
			normalize_vec3f(&B);
			
			float3 dir;
			float3 use_color = make_float3(1,1,1);
			float n;
			float omega_z;
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
				dir.z = powf(random.x, 1.0f/(n+1.0f));
				dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*random.y);
				dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*random.y);
				omega_z = dir.z;
				dir = transform_from_tangent_frame(dir, T, B, N);
				// store reflection ray differentials. (todo: change to broadened cone)
				ray_diff_dir[id] = reflect(upper_dir, N);
				ray_diff_dir[w*h+id] = reflect(right_dir, N);
				use_color = diffuse;
				float3 brdf = fabsf(dir|N) * diffuse * float(M_1_PI);// * fmaxf((N|-org_dir), 0.0f);
				TP *= brdf;
			}
			else if (specular_bounce) {
				make_tangent_frame(R, T, B);
				n = shininess;
				dir.z = powf(random.x, 1.0f/(n+1.0f));
				dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*random.y);
				dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*random.y);
				omega_z = dir.z;
				dir = transform_from_tangent_frame(dir, T, B, R);
				// store reflection ray differentials.
				ray_diff_dir[id] = reflect(upper_dir, N);
				ray_diff_dir[w*h+id] = reflect(right_dir, N);
				if ((dir|N)<0) specular_bounce = false;
				use_color = specular;
				float3 brdf = fabsf(dir|N) * (shininess + 1)*specular * 0.5 * M_1_PI * pow(fmaxf((R|dir), 0.0f), shininess);
				TP *= brdf;
			}
			if (diffuse_bounce||specular_bounce) {
				float len = length_of_vector(dir);
				dir /= len;

				P += 0.01*dir;
				ray_orig[id] = P;
				ray_dir[id]  = dir;
				max_t[id]    = FLT_MAX;
				TP *= (1.0f/P_component) * ((2.0f*float(M_PI))/((n+1.0f)*pow(omega_z,n)));
				throughput[id] = TP;
// 				throughput[id] = make_float3(0,0,0);
				return;
			}
		}
		// fall through for absorption and invalid intersection
		ray_dir[id]  = make_float3(0,0,0);
		ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		max_t[id] = -1;
		throughput[id] = make_float3(0,0,0);
	}

}

void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, halton_pool2f uniform_random, float3 *throughput, float3 *col_accum,
										  float *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, random_sampler_path_info pi) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::compute_path_contribution_and_bounce <<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  (float3*)ray_diff_org, (float3*)ray_diff_dir,
																  ti, triangles, mats, uniform_random, throughput, 
																  col_accum, (float3*)to_light, shadow_ti, potential_sample_contribution,
																  pi);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}

void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, lcg_random_state uniform_random, float3 *throughput, float3 *col_accum,
										  float *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, random_sampler_path_info pi) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::compute_path_contribution_and_bounce <<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  (float3*)ray_diff_org, (float3*)ray_diff_dir,
																  ti, triangles, mats, uniform_random, throughput, 
																  col_accum, (float3*)to_light, shadow_ti, potential_sample_contribution, 
																  pi);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}

void compute_path_contribution_and_bounce(int w, int h, float *ray_orig, float *ray_dir, float *max_t, float *ray_diff_org, float *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, mt_pool3f uniform_random, float3 *throughput, float3 *col_accum,
										  float *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, random_sampler_path_info pi) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::compute_path_contribution_and_bounce <<<blocks, threads>>>(w, h, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  (float3*)ray_diff_org, (float3*)ray_diff_dir,
																  ti, triangles, mats, uniform_random, throughput, 
																  col_accum, (float3*)to_light, shadow_ti, potential_sample_contribution, 
																  pi);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
}

