#include "material.h"

#include <librta/material.h>
#include <string>
#include <stdexcept>

using namespace std;

			
static size_t data_size = 0;

namespace rta {
	namespace cuda {
		bool verbose = true;
		int T;

		
		texture_data* convert_texture(rta::texture *t) {
			texture_data *new_tex = new cuda::texture_data(t->w, t->h, texture_data::device);
			// cpu rta uses float textures, this is to expensive on the gpu.
			unsigned char *data = new unsigned char[t->w*t->h*4];
			if (verbose) cout << "texture (" << T++ << ") " << t->filename << ": " << t->w << " x " << t->h << ": " << (t->w*t->h*6)/(1024*1024) << " MB" << endl;
			data_size += t->w*t->h*6;
			for (int y = 0; y < t->h; ++y)
				for (int x = 0; x < t->w; ++x) {
					vec3f c = t->sample(x/float(t->w),1.0f-y/float(t->h));
					data[4*(y*t->w+x)+0] = (unsigned char)(255.0f * c.x);
					data[4*(y*t->w+x)+1] = (unsigned char)(255.0f * c.y);
					data[4*(y*t->w+x)+2] = (unsigned char)(255.0f * c.z);
					data[4*(y*t->w+x)+3] = 255;
				}
			new_tex->upload(data);
			checked_cuda(cudaDeviceSynchronize());
			compute_mipmaps(new_tex);
			checked_cuda(cudaDeviceSynchronize());
			cuda::texture_data *gpu_tex;
			checked_cuda(cudaMalloc(&gpu_tex, sizeof(cuda::texture_data)));
			checked_cuda(cudaMemcpy(gpu_tex, new_tex, sizeof(cuda::texture_data), cudaMemcpyHostToDevice));
			delete [] data;
			return gpu_tex;
		}

		cuda::material_t* convert_and_upload_materials(int &N, std::vector<std::string> &subdFilenamesSet) {
			vector<rta::material_t*> coll;
			for (int i = 0; ; ++i)  {
				try {
					coll.push_back(rta::material(i));
				}
				catch (runtime_error &e) {
					break;
				}
			}
			//HACK: ADD DEFAULT PBRT MATERIAL: always last one!
			coll.push_back(rta::material(0));
			std::vector<std::string> subdFilenames;
//			for(auto it = subdFilenamesSet.begin(); it!=subdFilenamesSet.end(); ++it){
			for(int i=0; i<subdFilenamesSet.size(); i++){
				std::string materialPath("materials/");
				std::string materialEnd(".pbrdf");
				std::string searchFileName = materialPath + subdFilenamesSet[i] + materialEnd;
				std::ifstream in(searchFileName.c_str());
				if(in.is_open()){
					std::cerr<<"Could open "<<searchFileName<<"\n";
					subdFilenames.push_back(searchFileName);
					in.close();
				}else{
					std::cerr << "Could not find " << searchFileName << "\n";
				}
			}
			// add default material.
			std::string materialPath("materials/default.pbrdf");
			std::ifstream in(materialPath);
			if(!in.is_open()) std::cerr << "WARNING: Could not open Default PBRDF Material " << materialPath << "\n";
			else{
				subdFilenames.push_back(materialPath);
  				in.close();
			}


			N = coll.size() + subdFilenames.size();
			int numObjMaterials = coll.size();
			for(int i=0; i<subdFilenames.size(); i++) coll.push_back(rta::material(0));
			cuda::material_t *materials = new cuda::material_t[coll.size()];
			data_size = 0;
			T=0;
			int subdidx = 0;
			for (int i = 0; i < N; ++i) {
				if(i >= numObjMaterials){
					//subdFilenames!
					//std::string materialPath("materials/");
//					std::string testFile = materialPath + subdFilenames[subdidx] + std::string(".pbrdf");
//					std::ifstream in(testFile.c_str());
					cuda::material_t *m = &materials[i];

				/*	if(in.is_open()){
						in.close();
						m->parameters = new PrincipledBRDFParameters(testFile);
					}else{
						std::string materialPath("materials/default.pbrdf");
						std::ifstream in(materialPath);
						if(!in.is_open()) std::cerr<<"WARNING: Could not open PBRDF Material "<<materialPath<<"\n";
						else in.close();
						m->parameters = new PrincipledBRDFParameters(materialPath);
					}*/
					m->parameters = new PrincipledBRDFParameters(subdFilenames[subdidx]);
					std::cerr<< "SETTING material "<<i<<" to subd "<<subdidx<<" to subd filename "<<subdFilenames[subdidx]<<"\n";
					subdidx++;
					continue;
				}
				rta::material_t *src = coll[i];
/*				if(i ==coll.size()-1){
					std::string materialPath("materials/default.pbrdf");
					std::ifstream in(materialPath.c_str());		
					cuda::material_t *m = &materials[i];
					if(in.is_open()){
						in.close();
						m->parameters = new PrincipledBRDFParameters(materialPath);
					}else{
						m->parameters = 0;
					}
					continue; // equivalent with break, since this is the last material element.
				}*/
				std::string materialPath ("materials/");
				std::string pbrdfEnding (".pbrdf");		
				std::string defaultMaterial = materialPath + src->name + pbrdfEnding;
				//a bit hacky. if there is a material in ./materials/ with the current material filename + .pbrdf ending
				//we use the parameters defined in the file and use the principled BRDF material evaluation for that object.
				std::ifstream in(defaultMaterial.c_str());
				bool useDefaultMaterial = (!in.is_open());
				cuda::material_t *m = &materials[i];
				if(useDefaultMaterial) m->parameters = 0;
				else{ 
					cout << "Found pbrdf material -> " << defaultMaterial << endl;
					in.close(); 
					m->parameters = new PrincipledBRDFParameters(defaultMaterial);
				}
				m->diffuse_color.x = src->diffuse_color.x;
				m->diffuse_color.y = src->diffuse_color.y;
				m->diffuse_color.z = src->diffuse_color.z;
				m->specular_color.x = src->specular_color.x;
				m->specular_color.y = src->specular_color.y;
				m->specular_color.z = src->specular_color.z;
				if (src->diffuse_texture)
					m->diffuse_texture = convert_texture(src->diffuse_texture);
				if (src->specular_texture)
					m->specular_texture = convert_texture(src->specular_texture);
				if (src->alpha_texture)
					m->alpha_texture = convert_texture(src->alpha_texture);
				m->alpha = src->alpha;
			}
			cuda::material_t *gpu_mats;
			checked_cuda(cudaMalloc(&gpu_mats, coll.size()*sizeof(cuda::material_t)));
			checked_cuda(cudaMemcpy(gpu_mats, materials, sizeof(cuda::material_t)*coll.size(), cudaMemcpyHostToDevice));
			delete [] materials;
			cout << coll.size() << " materials (" << data_size/(1024*1024) << "MiB)." << endl;
			return gpu_mats;
		}

		cuda::texture_data* download_texture(cuda::texture_data *gpu) {
			cuda::texture_data *tex = new cuda::texture_data;
			checked_cuda(cudaMemcpy(tex, gpu, sizeof(cuda::texture_data), cudaMemcpyDeviceToHost));
			tex->location = cuda::texture_data::host;
			unsigned char *gpu_data = tex->rgba;
			tex->rgba = new unsigned char[tex->w * tex->h * 6];
			checked_cuda(cudaMemcpy(tex->rgba, gpu_data, sizeof(unsigned char)*6*tex->w*tex->h, cudaMemcpyDeviceToHost));
			cout << "converted tex " << tex->w << " x " << tex->h << ", mm=" << tex->max_mm << " on " << tex->location << endl;
			return tex;
		}

		cuda::material_t* download_materials(cuda::material_t *gpu_mats, int nr_of_materials) {
			cuda::material_t *materials = new cuda::material_t[nr_of_materials];
			checked_cuda(cudaMemcpy(materials, gpu_mats, sizeof(cuda::material_t)*nr_of_materials, cudaMemcpyDeviceToHost));
			for (int i = 0; i < nr_of_materials; ++i) {
				auto &mtl = materials[i];
				if (mtl.diffuse_texture)  mtl.diffuse_texture  = download_texture(mtl.diffuse_texture);
				if (mtl.specular_texture) mtl.specular_texture = download_texture(mtl.specular_texture);
				if (mtl.alpha_texture)    mtl.alpha_texture    = download_texture(mtl.alpha_texture);
			}
			return materials;
		}

	}
}

/* vim: set foldmethod=marker: */

