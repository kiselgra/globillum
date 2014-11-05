#include "gpu_cgls_lights.h"

#include "material.h"
#include "cgls-lights.h"
#include "vars.h"

#include <libhyb/trav-util.h>

using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;

namespace local {

// 	vec3f *material;

	template<typename _box_t, typename _tri_t> struct gpu_material_evaluator : public cuda::gpu_ray_bouncer<forward_traits> {
		declare_traits_types;
		cuda::material_t *materials;
		cuda::simple_triangle *tri_ptr;
		float3 *material_colors;
		cuda::cam_ray_generator_shirley *crgs;
		gpu_material_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, cuda::cam_ray_generator_shirley *crgs)
			: cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), material_colors(0), tri_ptr(triangles),
			  crgs(crgs) {
			checked_cuda(cudaMalloc(&material_colors, sizeof(float3)*w*h));
		}
		~gpu_material_evaluator() {
			checked_cuda(cudaFree(material_colors));
		}
		virtual bool trace_further_bounces() {
			return false;
		}
		virtual void bounce() {
			rta::cuda::evaluate_material(this->w, this->h, this->gpu_last_intersection, tri_ptr, materials, material_colors, crgs->gpu_origin, crgs->gpu_direction);
		}
		virtual std::string identification() {
			return "evaluate first-hit material on gpu.";
		}
	};

	template<typename _box_t, typename _tri_t> struct gpu_cgls_light_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		cuda::cgls::light *lights;
		int nr_of_lights;
		gpu_cgls_light_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, cuda::cam_ray_generator_shirley *crgs, cuda::cgls::light *lights, int nr_of_lights)
		: gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs), lights(lights), nr_of_lights(nr_of_lights) {
		}
		virtual void bounce() {
			gpu_material_evaluator<forward_traits>::bounce();
			rta::cuda::cgls::add_shading(this->w, this->h, this->material_colors, lights, nr_of_lights, this->gpu_last_intersection, this->tri_ptr);
		}
		virtual std::string identification() {
			return "evaluate first-hit material and shade with cgls lights.";
		}
	};

	void gpu_cgls_lights::activate(rt_set *orig_set) {
		set = *orig_set;
		set.rt = set.rt->copy();
		int nr_of_gpu_lights;
		cuda::cgls::light *gpu_lights = cuda::cgls::convert_and_upload_lights(scene, nr_of_gpu_lights);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::cam_ray_generator_shirley(w, h);
// 		set.bouncer = new gpu_material_evaluator<B, T>(w, h, gpu_materials, triangles, crgs);
		set.bouncer = new gpu_cgls_light_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights);
		set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
		set.basic_rt<B, T>()->ray_generator(set.rgen);
	}

	void gpu_cgls_lights::compute() {
			cout << "tracing..." << endl;
			vec3f pos, dir, up;
			matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
			extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
			extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
			extract_up_vec3f_of_matrix(&up, lookat_matrix);
			crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));
			set.rt->trace();

			cout << "primary visibility took " << set.basic_rt<B, T>()->timings.front() << "ms." << endl;
	
			cout << "saving output" << endl;
			png::image<png::rgb_pixel> image(w, h);
			vec3f *data = new vec3f[w*h];
			cudaMemcpy(data, (dynamic_cast<gpu_material_evaluator<B, T>*>(set.bouncer))->material_colors, sizeof(float3)*w*h, cudaMemcpyDeviceToHost);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					vec3f *pixel = data+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(255*pixel->x,0,255), clamp(255*pixel->y,0,255), clamp(255*pixel->z,0,255))); 
				}
			}
			image.write("out.png");
	}
		


	/*
		void gpu_cgls_lights::save(vec3f *out) {
			cout << "saving output" << endl;
			png::image<png::rgb_pixel> image(w, h);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					vec3f *pixel = out+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(255*pixel->x,0,255), clamp(255*pixel->y,0,255), clamp(255*pixel->z,0,255))); 
				}
			}
			image.write("out.png");
		}
		*/

}

/* vim: set foldmethod=marker: */

