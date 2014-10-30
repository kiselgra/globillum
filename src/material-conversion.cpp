#include "material.h"

#include <librta/material.h>

#include <string>
#include <stdexcept>

using namespace std;

namespace rta {
	namespace cuda {

		cuda::material_t* convert_and_upload_materials() {
			vector<rta::material_t*> coll;
			for (int i = 0; ; ++i)  {
				try {
					coll.push_back(rta::material(i));
				}
				catch (runtime_error &e) {
					break;
				}
			}
			cuda::material_t *materials = new cuda::material_t[coll.size()];
			for (int i = 0; i < coll.size(); ++i) {
				rta::material_t *src = coll[i];
				cuda::material_t *m = &materials[i];
				m->diffuse_color.x = src->diffuse_color.x;
				m->diffuse_color.y = src->diffuse_color.y;
				m->diffuse_color.z = src->diffuse_color.z;
				if (src->diffuse_texture) {
					rta::texture *t = src->diffuse_texture;
					m->diffuse_texture = new cuda::texture_data(t->w, t->h);
					// cpu rta uses float textures, this is to expensive on the gpu.
					unsigned char *data = new unsigned char[t->w*t->h*4];
					for (int y = 0; y < t->h; ++y)
						for (int x = 0; x < t->w; ++x) {
							vec3f c = t->sample(x/float(t->w),1.0f-y/float(t->h));
							data[4*(y*t->w+x)+0] = (unsigned char)(255.0f * c.x);
							data[4*(y*t->w+x)+1] = (unsigned char)(255.0f * c.y);
							data[4*(y*t->w+x)+2] = (unsigned char)(255.0f * c.z);
							data[4*(y*t->w+x)+3] = 255;
						}
					m->diffuse_texture->upload(data);
					checked_cuda(cudaDeviceSynchronize());
					compute_mipmaps(m->diffuse_texture);
					checked_cuda(cudaDeviceSynchronize());
					cuda::texture_data *gpu_tex;
					checked_cuda(cudaMalloc(&gpu_tex, sizeof(cuda::texture_data)));
					checked_cuda(cudaMemcpy(gpu_tex, m->diffuse_texture, sizeof(cuda::texture_data), cudaMemcpyHostToDevice));
					delete m->diffuse_texture;
					m->diffuse_texture = gpu_tex;
					delete [] data;
				}
			}
			cuda::material_t *gpu_mats;
			checked_cuda(cudaMalloc(&gpu_mats, coll.size()*sizeof(cuda::material_t)));
			checked_cuda(cudaMemcpy(gpu_mats, materials, sizeof(cuda::material_t)*coll.size(), cudaMemcpyHostToDevice));
			delete [] materials;
			cout << coll.size() << " materials." << endl;
			return gpu_mats;
		}

	}
}

/* vim: set foldmethod=marker: */

