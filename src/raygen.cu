#include "raygen.h"

#include <librta/basic_types.h>
#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace rta {
	namespace cuda {
		namespace k {
			__global__ void setup_jittered_shirley(float *dirs, float *orgs, float *maxts, 
												   float fovy, float aspect, int w, int h, float3 view_dir, float3 pos, float3 up, float maxt,
												   gi::cuda::mt_pool3f uniform_random_01) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h)
					return;
				
				maxts[gid.y * w + gid.x] = maxt;
				fovy /= 2.0;
				float height = tanf(M_PI * fovy / 180.0f);
				float width = aspect * height;

				float3 random = gi::next_random3f(uniform_random_01, gid.y*w+gid.x);
				float jx = float(gid.x) + random.x;
				float jy = float(gid.y) + random.y;
				
// 				float u_s = (((float)gid.x+0.5f)/(float)w) * 2.0f - 1.0f;	// \in (-1,1)
// 				float v_s = (((float)gid.y+0.5f)/(float)h) * 2.0f - 1.0f;
				float u_s = ((jx)/(float)w) * 2.0f - 1.0f;	// \in (-1,1)
				float v_s = ((jy)/(float)h) * 2.0f - 1.0f;
				u_s = width * u_s;	// \in (-pw/2, pw/2)
				v_s = height * v_s;
				
				float3 vd = view_dir;
				float3 vu = up;
				float3 W, TxW, U, V;
				div_vec3f_by_scalar(&W, &vd, length_of_vec3f(&vd));
				cross_vec3f(&TxW, &vu, &W);
				div_vec3f_by_scalar(&U, &TxW, length_of_vec3f(&TxW));
				cross_vec3f(&V, &W, &U);

				float3 dir = make_float3(0,0,0), tmp;
				mul_vec3f_by_scalar(&dir, &U, u_s);
				mul_vec3f_by_scalar(&tmp, &V, v_s);
				add_components_vec3f(&dir, &dir, &tmp);
				add_components_vec3f(&dir, &dir, &W);
				normalize_vec3f(&dir);
				
// 				if (gid.x == 200 && gid.y == 100) {
// 					printf("dir %6.6f %6.6f %6.6f %f\n", dir.x, dir.y, dir.z, (dir|view_dir));
// 					printf("pos %6.6f %6.6f %6.6f\n", pos.x, pos.y, pos.z);
// 				}
					
				dirs[3*(gid.y * w + gid.x)+0] = dir.x;
				dirs[3*(gid.y * w + gid.x)+1] = dir.y;
				dirs[3*(gid.y * w + gid.x)+2] = dir.z;
				orgs[3*(gid.y * w + gid.x)+0] = pos.x;
				orgs[3*(gid.y * w + gid.x)+1] = pos.y;
				orgs[3*(gid.y * w + gid.x)+2] = pos.z;
			}
				
			__global__ void setup_jittered_dof_shirley(float *dirs, float *orgs, float *maxts, 
													   float fovy, float aspect, int w, int h, float3 view_dir, float3 pos, float3 up, float maxt,
													   float focus_distance, float aperture, float eye_to_lens,
													   gi::cuda::mt_pool3f uniform_random_01) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h)
					return;
				
				int id = gid.y * w + gid.x;
				maxts[id] = maxt;
				fovy /= 2.0;
				float height = tanf(M_PI * fovy / 180.0f);
				float width = aspect * height;

				float3 random = gi::next_random3f(uniform_random_01, id);
				float jx = float(gid.x) + random.x;
				float jy = float(gid.y) + random.y;
				
				float u_s = ((jx)/(float)w) * 2.0f - 1.0f;	// \in (-1,1)
				float v_s = ((jy)/(float)h) * 2.0f - 1.0f;
				u_s = width * u_s;	// \in (-pw/2, pw/2)
				v_s = height * v_s;
				
				float3 vd = view_dir;
				float3 vu = up;
				float3 W, TxW, U, V;
				div_vec3f_by_scalar(&W, &vd, length_of_vec3f(&vd));
				cross_vec3f(&TxW, &vu, &W);
				div_vec3f_by_scalar(&U, &TxW, length_of_vec3f(&TxW));
				cross_vec3f(&V, &W, &U);

				float3 dir = make_float3(0,0,0), tmp;
				mul_vec3f_by_scalar(&dir, &U, u_s);
				mul_vec3f_by_scalar(&tmp, &V, v_s);
				add_components_vec3f(&dir, &dir, &tmp);
				add_components_vec3f(&dir, &dir, &W);
				normalize_vec3f(&dir);
				
				float3 pos_on_focal_plane = pos + dir*(1.0f/(dir|view_dir))*focus_distance;
				float2 jitter = make_float2(0,0);
				int i=1;
				do {
					random = gi::next_random3f(uniform_random_01, (id+17*i)%(w*h));
					jitter = make_float2(random.z-0.5f, random.y-0.5f);
					if (i == 100) { jitter.x = jitter.y = 0; break; }
				} while (jitter.x*jitter.x + jitter.y*jitter.y > 1.0f);

				float3 jitter_pos = pos + U*jitter.x*aperture + V*jitter.y*aperture;
				dir = (pos_on_focal_plane - jitter_pos);
				normalize_vec3f(&dir);

				dirs[3*(id)+0] = dir.x;
				dirs[3*(id)+1] = dir.y;
				dirs[3*(id)+2] = dir.z;
				orgs[3*(id)+0] = jitter_pos.x;
				orgs[3*(id)+1] = jitter_pos.y;
				orgs[3*(id)+2] = jitter_pos.z;
			}
		}
			
		void setup_jittered_shirley(float *dirs, float *orgs, float *maxts, 
									float fovy, float aspect, int w, int h, float3 *view_dir, float3 *pos, float3 *up, float maxt,
									gi::cuda::mt_pool3f uniform_random_01) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::setup_jittered_shirley<<<blocks, threads>>>(dirs, orgs, maxts, fovy, aspect, w, h, *view_dir, *pos, *up, maxt,
															   uniform_random_01);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
		}
			
		void setup_jittered_lens_shirley(float *dirs, float *orgs, float *maxts, 
										 float fovy, float aspect, int w, int h, float3 *view_dir, float3 *pos, float3 *up, float maxt,
										 float focus_distance, float aperture, float eye_to_lens,
										 gi::cuda::mt_pool3f uniform_random_01) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				std::cout << "RG: " << aperture << ", " << focus_distance << ", " << eye_to_lens << std::endl;
				k::setup_jittered_dof_shirley<<<blocks, threads>>>(dirs, orgs, maxts, fovy, aspect, w, h, *view_dir, *pos, *up, maxt,
																   focus_distance, aperture, eye_to_lens, uniform_random_01);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
		}
	
	}
}
