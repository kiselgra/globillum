#include "cgls-lights.h"

#include <libcgls/light.h>
#include <libhyb/trav-util.h>

#include <vector>
#include <stdexcept>

using namespace std;

namespace rta {
	namespace cuda {
		namespace cgls {

			static float3 conv(const vec3f &v) { return make_float3(v.x, v.y, v.z); }

			light* convert_and_upload_lights(scene_ref scene, int &N) {
				N = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					int t = light_type(run->ref);
					if (t == hemi_light_t || t == spot_light_t)
						++N;
				}
				light *L;
				cout << "lights: " << N << endl;
				checked_cuda(cudaMalloc(&L, sizeof(light)*N));
				update_lights(scene, L, N);
				return L;
			}

			void update_lights(scene_ref scene, light *data, int N) {
				vector<light> lights;
				int n = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next) {
					light l;
					vec3f dir, pos;
					extract_pos_vec3f_of_matrix(&pos, light_trafo(run->ref));
					extract_dir_vec3f_of_matrix(&dir, light_trafo(run->ref));
					l.pos = conv(pos);
					l.dir = conv(dir);
					l.col = conv(*light_color(run->ref));
					int t = light_type(run->ref);
					if (t == hemi_light_t) {
						l.type = light::hemi;
						l.dir = *(float3*)light_aux(run->ref);
					}
					else if (t == spot_light_t) {
						l.type = light::spot;
						l.spot_cos_cutoff = cosf(*(float*)light_aux(run->ref));
					}
					else {
// 						cerr << "ignoring light '" << light_name(run->ref) << "' because of incompatible type." << endl;
						continue;
					}
					lights.push_back(l);
					++n;
				}
				if (n != N)
					throw std::runtime_error("number of lights changed in " "update_lights");
				checked_cuda(cudaMemcpy(data, &lights[0], sizeof(light)*n, cudaMemcpyHostToDevice));
			}



			namespace k {
				__global__ void add_shading(int w, int h, float3 *color_data, light *lights, int nr_of_lights,
											triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
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
										out += factor * material_color ;//* lights[i].col;
									}
								}
							}
						}
					}

					color_data[gid.y*w+gid.x] = out;
				}
			}

			void add_shading(int w, int h, float3 *color_data, light *lights, int nr_of_lights, 
							 triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::add_shading<<<blocks, threads>>>(w, h, color_data, lights, nr_of_lights, ti, triangles);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}

		}
	}
}
