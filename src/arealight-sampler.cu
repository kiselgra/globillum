#include "arealight-sampler.h"

#include "util.h"

#include <libhyb/trav-util.h>

#include <vector>
#include <stdexcept>
#include <iostream>

#include <cuda_gl_interop.h>


#define USE_SKYLIGHT_SAMPLING

using namespace std;
using namespace rta;
using namespace gi;

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


		// 
		// Setup new area light samples for direct lighting computations.
		//
			
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
// 					float3 right = make_float3(1,0,0);
// 					float3 up = make_float3(0,0,1);
					float3 rnd = next_random3f(uniform01, id);
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
			
			template<typename rng_t> __host__ __device__ __forceinline__
			void pixel_generate_arealight_sample(int2 gid, 
												 int w, int h, gi::light *lights, int nr_of_lights, float overall_power,
												 float3 *ray_orig, float3 *ray_dir, float *max_t,
												 triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
												 rng_t uniform01, float3 *potential_sample_contribution) {
				int id = gid.y*w+gid.x;
				float3 rnd = gi::next_random3f(uniform01, id);
				float choice = rnd.z*overall_power;
				float light_acc = 0;
				int light = 0;
				while (choice > light_acc+lights[light].power && light < nr_of_lights) {
					light_acc += lights[light].power;
					++light;
				}
				
				triangle_intersection<cuda::simple_triangle> is = ti[id];
				if (is.valid()) {
					float3 bc; 
					float3 P, N;
					if ((is.ref & 0xff000000) == 0) {
						cuda::simple_triangle tri = triangles[is.ref];
						is.barycentric_coord(&bc);
						barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
						barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
					}
					else {
// 						unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
// 						unsigned int ptexID = 0x00ffffff & is.ref;
// 						bool WITH_DISPLACEMENT = true;
// 						if (WITH_DISPLACEMENT)
// 							models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, true, (float*)&P, (float*)&N);
// 						else
// 							models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, false, (float*)&P, (float*)&N);
					}

					float3 contribution = make_float3(0.0f,0.0f,0.0f);
					float3 dir;
					float len;

					if (lights[light].type == gi::light::rect) {
						float3 light_pos = lights[light].rectlight.center;
						float3 light_dir = lights[light].rectlight.dir;
						float3 right = make_tangential(make_float3(1,0,0), light_dir);
						float3 up = make_tangential(make_float3(0,1,0), light_dir);
						// 						float3 right = make_float3(1,0,0);
						// 						float3 up = make_float3(0,0,1);
						float2 offset = make_float2((rnd.x - 0.5f) * lights[light].rectlight.wh.x,
													(rnd.y - 0.5f) * lights[light].rectlight.wh.y);
						float3 light_sample = light_pos + offset.x * right + offset.y * up;
						dir = light_sample - P;
						len = length_of_vector(dir);
						dir /= len;
						float ndotl = fmaxf((N|dir), 0.0f);
						float light_cos = fmaxf((light_dir|-dir), 0.0f);
						float factor = lights[light].rectlight.wh.x * lights[light].rectlight.wh.y * ndotl * light_cos / (len*len);
						contribution = lights[light].rectlight.col * factor;
					}
					else if (lights[light].type == gi::light::sky) {
						len = FLT_MAX;
						sky_light &sl = lights[light].skylight;
					#ifdef USE_SKYLIGHT_SAMPLING
						float outPdf = 1.0f;
						float3 L = sl.sample(rnd.x, rnd.y, outPdf, dir);
						dir = make_tangential(dir,N);
						float a = 1.0f/(outPdf);	
						contribution = sl.scale * L * a * fabs(dir|N);
					#else
						float sq = sqrtf(1-rnd.x*rnd.x);
                                                dir.x = sq * cosf(2.0f*float(M_PI)*rnd.y);
                                                dir.y = sq * sinf(2.0f*float(M_PI)*rnd.y);
                                                dir.z = rnd.x;
                                                dir = make_tangential(dir, N);
						float theta = acosf(dir.y);
						float phi = atan2f(dir.z, dir.x);
						if (phi < 0) phi += 2.0f*float(M_PI);
						float s = phi/(2.0f*float(M_PI));
						float t = theta/float(M_PI);
						contribution = sl.scale * sl.data[int(t*sl.h) * sl.w + int(s*sl.w)] * (dir|N);
					#endif						

					}
					
					P += 0.01*dir;
					ray_orig[id] = P;
					ray_dir[id]  = dir;
					max_t[id]    = len;
					potential_sample_contribution[id] = contribution * (overall_power/lights[light].power);
				}
				else {
					float3 dir = ray_dir[id];
					ray_dir[id]  = make_float3(0,0,0);
					ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
					max_t[id] = -1;
					// potential_sample_contribution[id] = make_float3(0, 0, 0);
					float3 Le = make_float3(0.0f,0.0f,0.0f);
					int i = 0;
					for (i = nr_of_lights-1; i >= 0; i--)
						if (lights[i].type == gi::light::sky)
							break;
					if (i >= 0) {
						float theta = acosf(dir.y);
						float phi = atan2f(dir.z, dir.x);
						if (phi < 0) phi += 2.0f*float(M_PI);
						float s = phi/(2.0f*float(M_PI));
						float t = theta/float(M_PI);
						sky_light &sl = lights[i].skylight;
						Le = sl.scale * sl.data[int(t*sl.h) * sl.w + int(s*sl.w)];
					}
					potential_sample_contribution[id] = Le;
				}
			}
				
			template<typename rng_t> __global__ 
			void generate_arealight_sample(int w, int h, gi::light *lights, int nr_of_lights, float overall_power,
										   float3 *ray_orig, float3 *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   rng_t uniform01, float3 *potential_sample_contribution, gi::cuda::random_sampler_path_info pi) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				pixel_generate_arealight_sample(gid,
												w, h, lights, nr_of_lights, overall_power, ray_orig, ray_dir, max_t,
												ti, triangles, uniform01, potential_sample_contribution);
					
			}
		}

			void generate_rectlight_sample(int w, int h, gi::light *lights, int nr_of_lights, float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::halton_pool2f uniform01, float3 *potential_sample_contribution, 
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
										   gi::lcg_random_state uniform01, float3 *potential_sample_contribution, 
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

			void generate_arealight_sample(int w, int h, gi::light *lights, int nr_of_lights, float overall_power,
										   float *ray_orig, float *ray_dir, float *max_t,
										   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
										   gi::cuda::mt_pool3f uniform01, float3 *potential_sample_contribution, 
										   gi::cuda::random_sampler_path_info pi) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::generate_arealight_sample<<<blocks, threads>>>(w, h, lights, nr_of_lights, overall_power, (float3*)ray_orig, (float3*)ray_dir, max_t, 
																  ti, triangles, uniform01, potential_sample_contribution, pi);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}

		}
			
			
	
		// 
		// Add light samples to direct lighting accumulation buffer
		//

		namespace cuda {

			namespace k {
				
				__host__ __device__ __forceinline__
				void pixel_integrate_light_sample(int2 gid,
												  int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
												  float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, float sample) {
									int id = gid.y*w+gid.x;
					float3 weight = potential_sample_contribution[id];
					triangle_intersection<cuda::simple_triangle> is = ti[id];
					// use material color if no intersection is found (ie the path to the light is clear).
					float3 material = make_float3(0,0,0);
					if (!is.valid())
						material = material_col[id];
					// use accum color if we should not clear
					float3 out = make_float3(0,0,0);
// 					if (is.valid()) out = make_float3(0,0.3,0);
// 					else out = make_float3(0,0,.3);
// 					if (gid.x == 600 && gid.y == 100)
// 						if (is.valid()) out = make_float3(1,0,0);
// 						else out = make_float3(1,1,0);
					if (sample > 0)
						out = col_accum[id];
					// out we go.
					out = (sample * out + weight * material) / (sample+1.0f);
					col_accum[id] = out;
				}

				__global__ void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
													   float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, float sample) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					pixel_integrate_light_sample(gid,
												 w, h, ti, potential_sample_contribution, material_col, col_accum, sample);
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
			
		}
			
		void integrate_light_sample(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
									float3 *potential_sample_contribution, float3 *material_col, float3 *col_accum, int sample) {
			#pragma omp parallel for 
			for (int y = 0; y < h; ++y) {
				for (int x = 0; x < w; ++x) {
					cuda::k::pixel_integrate_light_sample(make_int2(x, y), 
														  w, h, ti, potential_sample_contribution, material_col, col_accum, float(sample));
				}
			}
		}


}


