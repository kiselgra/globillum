#include "arealight-sampler.h"

#include "util.h"

#include <libhyb/trav-util.h>

#include <vector>
#include <stdexcept>
#include <iostream>

#include <cuda_gl_interop.h>

using namespace std;

namespace rta {
	namespace cuda {
		namespace cgls {
			
			static float3 conv(const vec3f &v) { return make_float3(v.x, v.y, v.z); }

			gi::light* convert_and_upload_rectangular_area_lights(scene_ref scene, int &N) {
				N = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					int t = light_type(run->ref);
					if (t == rect_light_t)
						++N;
				}
				gi::light *L;
				checked_cuda(cudaMalloc(&L, sizeof(gi::light)*N));
				update_rectangular_area_lights(scene, L, N);
				return L;
			}

			void update_rectangular_area_lights(scene_ref scene, gi::light *data, int N) {
				vector<gi::light> lights;
				int n = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					int t = light_type(run->ref);
					if (t == rect_light_t) {
						gi::light l;
						vec3f dir, pos, up;
						extract_pos_vec3f_of_matrix(&pos, light_trafo(run->ref));
						extract_dir_vec3f_of_matrix(&dir, light_trafo(run->ref));
						extract_up_vec3f_of_matrix(&up, light_trafo(run->ref));
						l.type = gi::light::rect;
						l.rectlight.center = conv(pos);
						l.rectlight.dir = conv(dir);
						l.rectlight.up = conv(up);
						l.rectlight.col = conv(*light_color(run->ref));
						l.rectlight.wh = *(float2*)light_aux(run->ref);
						lights.push_back(l);
						++n;
					}
				}
				if (n != N)
					throw std::runtime_error("number of lights changed in " "update_rectangular_area_lights");
				checked_cuda(cudaMemcpy(data, &lights[0], sizeof(gi::light)*n, cudaMemcpyHostToDevice));
			}
				
			namespace k {
				template<typename rng_t> __global__ 
				void generate_rectlight_sample(int w, int h, gi::light *lights, int nr_of_lights, float3 *ray_orig, float3 *ray_dir, float *max_t,
											   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
											   rng_t uniform01, float3 *potential_sample_contribution, gi::cuda::random_sampler_path_info pi) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					int id = gid.y*w+gid.x;
					
					triangle_intersection<cuda::simple_triangle> is = ti[id];
					if (is.valid()) {
						float3 bc; 
						float3 P, N;
						cuda::simple_triangle tri = triangles[is.ref];
						is.barycentric_coord(&bc);
						barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
						barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
						float3 light_pos = lights[0].rectlight.center;
						float3 light_dir = lights[0].rectlight.dir;
						float3 right = make_tangential(make_float3(1,0,0), light_dir);
						float3 up = make_tangential(make_float3(0,1,0), light_dir);
// 						float3 right = make_float3(1,0,0);
// 						float3 up = make_float3(0,0,1);
						float3 rnd = next_random3f(uniform01, id, pi);
						float2 offset = make_float2((rnd.x - 0.5f) * lights[0].rectlight.wh.x,
													(rnd.y - 0.5f) * lights[0].rectlight.wh.y);
						float3 light_sample = light_pos + offset.x * right + offset.y * up;
						float3 dir = light_sample - P;
						float len = length_of_vector(dir);
						dir /= len;
						P += 0.01*dir;
						ray_orig[id] = P;
						ray_dir[id]  = dir;
						max_t[id]    = len;
						float ndotl = fmaxf((N|dir), 0.0f);
						float light_cos = fmaxf((light_dir|-dir), 0.0f);
						float factor = lights[0].rectlight.wh.x * lights[0].rectlight.wh.y * ndotl * light_cos / (len*len);
						potential_sample_contribution[id] = lights[0].rectlight.col * factor;
					}
					else {
						ray_dir[id]  = make_float3(0,0,0);
						ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
						max_t[id] = -1;
						potential_sample_contribution[id] = make_float3(0, 0, 0);
					}
				}
			}

			void generate_rectlight_sample(int w, int h, gi::light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::halton_pool2f uniform01, float3 *potential_sample_contribution, 
										   gi::cuda::random_sampler_path_info pi) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::generate_rectlight_sample<<<blocks, threads>>>(w, h, lights, nr_of_lights, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  ti, triangles, uniform01, potential_sample_contribution, pi);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
			
			void generate_rectlight_sample(int w, int h, gi::light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::lcg_random_state uniform01, float3 *potential_sample_contribution, 
										   gi::cuda::random_sampler_path_info pi) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::generate_rectlight_sample<<<blocks, threads>>>(w, h, lights, nr_of_lights, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  ti, triangles, uniform01, potential_sample_contribution, pi);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}

			void generate_rectlight_sample(int w, int h, gi::light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::mt_pool3f uniform01, float3 *potential_sample_contribution, 
										   gi::cuda::random_sampler_path_info pi) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::generate_rectlight_sample<<<blocks, threads>>>(w, h, lights, nr_of_lights, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  ti, triangles, uniform01, potential_sample_contribution, pi);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}

			namespace k {
				__global__ void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
													   float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, float sample) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					int id = gid.y*w+gid.x;
					float3 weight = potential_sample_contribution[id];
					triangle_intersection<cuda::simple_triangle> is = ti[id];
					// use material color if no intersection is found (ie the path to the light is clear).
					float3 material = make_float3(0,0,0);
					if (!is.valid())
						material = material_col[id];
					// use accum color if we should not clear
					float3 out = make_float3(0,0,0);
					if (sample > 0)
						out = col_accum[id];
					// out we go.
					out = (sample * out + weight * material) / (sample+1.0f);
					col_accum[id] = out;
				}
				__global__ void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
													   float3 *potential_sample_contribution, float3 *material_col, float3 *throughput,
													   float3 *col_accum, float sample) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					int id = gid.y*w+gid.x;
					float3 weight = potential_sample_contribution[id];
					triangle_intersection<cuda::simple_triangle> is = ti[id];
					float3 tp = throughput[id];
					// use material color if no intersection is found (ie the path to the light is clear).
					float3 material = material_col[id];
					float3 use_material = material;
					if (is.valid())
						use_material = make_float3(0,0,0);
					// use accum color if we should not clear
					float3 out = make_float3(0,0,0);
					if (sample > 0)
						out = col_accum[id];
					// out we go.
					out = out + tp * use_material * weight;
					col_accum[id] = out;
					throughput[id] = tp * material * weight;
				}
			}

			void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
										float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, int sample) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::integrate_light_sample<<<blocks, threads>>>(w, h, ti, potential_sample_contribution, material_col, col_accum, float(sample));
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
			

			void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
										float3 *potential_sample_contribution, float3 *material_col, float3 *throughput, float3 *col_accum, int sample) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::integrate_light_sample<<<blocks, threads>>>(w, h, ti, potential_sample_contribution, material_col, throughput, col_accum, float(sample));
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
			

			// copy cuda colors to gl texture

			static cudaGraphicsResource *cuda_resource = 0;
			static cudaArray *cu_array = 0;
			static surface<void, 2> image_surf;

			namespace k {
				__global__ void copy_image_to_texture(int w, int h, float3 *colors, float scale) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					int id = gid.y*w+gid.x;
					float3 col = colors[id];
					surf2Dwrite(make_float4(col.x*scale, col.y*scale, col.z*scale, 1.0f), image_surf, gid.x*sizeof(float4), gid.y);
				}
			}
	
			void init_cuda_image_transfer(texture_ref tex) {
				checked_cuda(cudaGraphicsGLRegisterImage(&cuda_resource, texture_id(tex), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
			}

			void copy_cuda_image_to_texture(int w, int h, float3 *col, float scale) {
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaGraphicsMapResources(1, &cuda_resource, 0));
				checked_cuda(cudaGraphicsSubResourceGetMappedArray(&cu_array, cuda_resource, 0, 0));
				checked_cuda(cudaGetLastError());
				checked_cuda(cudaDeviceSynchronize());
				checked_cuda(cudaBindSurfaceToArray(image_surf, cu_array));

				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::copy_image_to_texture<<<blocks, threads>>>(w, h, col, scale);

				checked_cuda(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
	
		}
	}
}


