#include "arealight-sampler.h"

#include "util.h"

#include <vector>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace rta {
	namespace cuda {
		namespace cgls {
			
			static float3 conv(const vec3f &v) { return make_float3(v.x, v.y, v.z); }

			rect_light* convert_and_upload_rectangular_area_lights(scene_ref scene, int &N) {
				N = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					int t = light_type(run->ref);
					if (t == rect_light_t)
						++N;
				}
				rect_light *L;
				cout << "rect lights: " << N << endl;
				checked_cuda(cudaMalloc(&L, sizeof(rect_light)*N));
				update_rectangular_area_lights(scene, L, N);
				return L;
			}

			void update_rectangular_area_lights(scene_ref scene, rect_light *data, int N) {
				vector<rect_light> lights;
				int n = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					int t = light_type(run->ref);
					if (t == rect_light_t) {
						rect_light l;
						vec3f dir, pos, up;
						extract_pos_vec3f_of_matrix(&pos, light_trafo(run->ref));
						extract_dir_vec3f_of_matrix(&dir, light_trafo(run->ref));
						extract_up_vec3f_of_matrix(&up, light_trafo(run->ref));
						l.center = conv(pos);
						l.dir = conv(dir);
						l.up = conv(up);
						l.col = conv(*light_color(run->ref));
						lights.push_back(l);
						++n;
					}
				}
				if (n != N)
					throw std::runtime_error("number of lights changed in " "update_rectangular_area_lights");
				checked_cuda(cudaMemcpy(data, &lights[0], sizeof(rect_light)*n, cudaMemcpyHostToDevice));
			}
				
			namespace k {
				__global__ void generate_rectlight_sample(int w, int h, rect_light *lights, int nr_of_lights, float3 *ray_orig, float3 *ray_dir, float *max_t,
														  triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
														  gi::cuda::halton_pool2f uniform01, float3 *potential_sample_contribution) {
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
						float3 light_pos = lights[0].center;
						float3 dir = light_pos - P;
						float len = length_of_vector(dir);
						dir /= len;
						P += 0.01*dir;
						ray_orig[id] = P;
						ray_dir[id]  = dir;
						max_t[id]    = len;
						float ndotl = fmaxf((N|dir), 0.0f);
						potential_sample_contribution[id] = make_float3(ndotl, ndotl, ndotl);
					}
					else {
						ray_dir[id]  = make_float3(0,0,0);
						ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
						max_t[id] = -1;
						potential_sample_contribution[id] = make_float3(0, 0, 0);
					}


					/*
					float3 material_color = color_data[gid.y*w+gid.x];
					float3 out = make_float3(0,0,0);

					triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
					if (is.valid()) {
						float3 bc; 
						float3 P, N;
						cuda::simple_triangle tri = triangles[is.ref];
						is.barycentric_coord(&bc);
						barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
						barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);

						for (int i = 0; i < nr_of_lights; ++i) {
							if (lights[i].type == light::hemi) {
								float factor = 0.5f * (1.0f + (N | lights[i].dir));
								out += factor * material_color * lights[i].col;
							}
							else if (lights[i].type == light::spot) {
								float3 l = lights[i].pos - P;
								float d = length_of_vector(l);
								l /= d;
								float3 dir = lights[i].dir;
								dir /= length_of_vector(dir);
								float ndotl = fmaxf((N|l), 0.0f);
								if (ndotl > 0) {
									float cos_theta = (dir|-l);
									if (cos_theta > lights[i].spot_cos_cutoff) {
										float angle = acosf(cos_theta);
										float cutoff = acosf(lights[i].spot_cos_cutoff);
										float factor = ndotl * (1.0f - smoothstep(cutoff * .7, cutoff, angle));
										out += factor * material_color * lights[i].col;
									}
								}
							}
						}
					}

					color_data[gid.y*w+gid.x] = out;
					*/
				}
			}

			void generate_rectlight_sample(int w, int h, rect_light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::halton_pool2f uniform01, float3 *potential_sample_contribution) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::generate_rectlight_sample<<<blocks, threads>>>(w, h, lights, nr_of_lights, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  ti, triangles, uniform01, potential_sample_contribution);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
			
			namespace k {
				__global__ void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
													   float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, bool clear) {
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
					if (!clear)
						out = col_accum[id];
					// out we go.
					out += weight * material;
					col_accum[id] = out;
				}
			
				__global__ void normalize_light_samples(int w, int h, float3 *col_accum, int samples) {
									int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					int id = gid.y*w+gid.x;
					float3 col = col_accum[id];
					col /= float(samples);
					col_accum[id] = col;
				}
			}

			void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
										float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, bool clear) {
				printf("  (%d)\n", clear?1:0);
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::integrate_light_sample<<<blocks, threads>>>(w, h, ti, potential_sample_contribution, material_col, col_accum, clear);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
			
			void normalize_light_samples(int w, int h, float3 *col_accum, int samples) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::normalize_light_samples<<<blocks, threads>>>(w, h, col_accum, samples);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}
	
	
		}
	}
}
