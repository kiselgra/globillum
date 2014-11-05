#include "cgls-lights.h"

#include <libcgls/light.h>

#include <vector>
#include <stdexcept>

using namespace std;

namespace rta {
	namespace cuda {
		namespace cgls {

			float3 conv(const vec3f &v) { return make_float3(v.x, v.y, v.z); }

			light* convert_and_upload_lights(scene_ref scene, int &N) {
				N = 0;
				for (light_list *run = scene_lights(scene); run; run = run->next)
					++N;
				light *L;
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
					else throw std::runtime_error("unknown light type!");
					lights.push_back(l);
					++n;
				}
				if (n != N)
					throw std::runtime_error("number of lights changed in " "update_lights");
				checked_cuda(cudaMemcpy(data, &lights[0], sizeof(light)*n, cudaMemcpyHostToDevice));
			}



			namespace k {
				__global__ void add_shading(int w, int h, float3 *color_data, light *lights, int nr_of_lights) {
					int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
										 blockIdx.y * blockDim.y + threadIdx.y);
					if (gid.x >= w || gid.y >= h) return;
					color_data[gid.y*w+gid.x].x *= 1.2;
					color_data[gid.y*w+gid.x].y *= 1.2;
					color_data[gid.y*w+gid.x].z *= 1.2;
				}
			}

			void add_shading(int w, int h, float3 *color_data, light *lights, int nr_of_lights) {
				checked_cuda(cudaPeekAtLastError());
				dim3 threads(16, 16);
				dim3 blocks = block_configuration_2d(w, h, threads);
				k::add_shading<<<blocks, threads>>>(w, h, color_data, lights, nr_of_lights);
				checked_cuda(cudaPeekAtLastError());
				checked_cuda(cudaDeviceSynchronize());
			}

		}
	}
}
