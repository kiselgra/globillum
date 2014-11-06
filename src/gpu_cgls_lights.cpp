#include "gpu_cgls_lights.h"

#include "material.h"
#include "cgls-lights.h"
#include "arealight-sampler.h"
#include "util.h"
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
		virtual void evaluate_material() {
			rta::cuda::evaluate_material(this->w, this->h, this->gpu_last_intersection, tri_ptr, materials, material_colors, 
										 crgs->gpu_origin, crgs->gpu_direction);
		}
		virtual void bounce() {
			evaluate_material();
		}
		virtual std::string identification() {
			return "evaluate first-hit material on gpu.";
		}
	};

	template<typename _box_t, typename _tri_t> struct gpu_cgls_light_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		cuda::cgls::light *lights;
		int nr_of_lights;
		gpu_cgls_light_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, 
								 cuda::cam_ray_generator_shirley *crgs, cuda::cgls::light *lights, int nr_of_lights)
		: gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs), lights(lights), nr_of_lights(nr_of_lights) {
		}
		virtual void shade_locally() {
			rta::cuda::cgls::add_shading(this->w, this->h, this->material_colors, lights, nr_of_lights, this->gpu_last_intersection, this->tri_ptr);
		}
		virtual void bounce() {
			this->evaluate_material();
			shade_locally();
		}
		virtual std::string identification() {
			return "evaluate first-hit material and shade with cgls lights.";
		}
	};
	
	template<typename _box_t, typename _tri_t> struct gpu_cgls_arealight_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		cuda::cgls::rect_light *lights;
		int nr_of_lights;
		gi::cuda::halton_pool2f uniform_random_numbers;
		float3 *potential_sample_contribution; 
		int curr_bounce;
		int samples;
		float3 *output_color;
		triangle_intersection<cuda::simple_triangle> *primary_intersection;
		gpu_cgls_arealight_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, 
									 cuda::cam_ray_generator_shirley *crgs, cuda::cgls::rect_light *lights, int nr_of_lights,
									 gi::cuda::halton_pool2f rnd, int samples)
		: gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs), 
		  lights(lights), nr_of_lights(nr_of_lights), uniform_random_numbers(rnd), samples(samples) {
			  checked_cuda(cudaMalloc(&potential_sample_contribution, sizeof(float3)*w*h));
			  checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
			  checked_cuda(cudaMalloc(&primary_intersection, sizeof(triangle_intersection<cuda::simple_triangle>)*w*h));
			  curr_bounce = 0;
		}
		~gpu_cgls_arealight_evaluator() {
			checked_cuda(cudaFree(potential_sample_contribution));
			checked_cuda(cudaFree(output_color));
			checked_cuda(cudaFree(primary_intersection));
		}
		virtual void new_pass() {
			curr_bounce = 0;
		}
		virtual bool trace_further_bounces() {
			return (curr_bounce < samples+1);
		}
		virtual void setup_new_arealight_sample() {
			rta::cuda::cgls::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
													   this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
													   primary_intersection, this->tri_ptr, uniform_random_numbers, potential_sample_contribution);
		}
		virtual void integrate_light_sample() {
			rta::cuda::cgls::integrate_light_sample(this->w, this->h, this->gpu_last_intersection, potential_sample_contribution,
													this->material_colors, output_color, curr_bounce==1);
		}
		virtual void normalize_output_color() {
			rta::cuda::cgls::normalize_light_samples(this->w, this->h, output_color, samples);
		}
		virtual void bounce() {
			cout << "bounce " << curr_bounce << endl;
			if (curr_bounce == 0) {
				cout << "eval mat" << endl;
				// this has to be done before switching the intersection data.
				this->evaluate_material();
				cout << "switch intersection data" << endl;
				triangle_intersection<cuda::simple_triangle> *tmp = primary_intersection;
				primary_intersection = this->gpu_last_intersection;
				this->gpu_last_intersection = tmp;
				cout << "setup samples" << endl;
				setup_new_arealight_sample();
			}
			else if (curr_bounce > 0) {
				cout << "integrate samples" << endl;
				integrate_light_sample();
				if (curr_bounce != samples) {
					cout << "setup samples" << endl;
					setup_new_arealight_sample();
				}
				else {
					cout << "normalize" << endl;
					normalize_output_color();
				}
			}
			++curr_bounce;
		}
		virtual std::string identification() {
			return "evaluate first-hit material and shade with cgls lights.";
		}
	};





	void gpu_cgls_lights::activate(rt_set *orig_set) {
		set = *orig_set;
		set.rt = set.rt->copy();
		gpu_lights = cuda::cgls::convert_and_upload_lights(scene, nr_of_gpu_lights);
		gpu_rect_lights = cuda::cgls::convert_and_upload_rectangular_area_lights(scene, nr_of_gpu_rect_lights);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::cam_ray_generator_shirley(w, h);
// 		set.bouncer = new gpu_material_evaluator<B, T>(w, h, gpu_materials, triangles, crgs);
// 		set.bouncer = new gpu_cgls_light_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights);
		gi::cuda::halton_pool2f pool = gi::cuda::generate_halton_pool_on_gpu(w*h);
		set.bouncer = new gpu_cgls_arealight_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_rect_lights, nr_of_gpu_rect_lights, pool, 16);
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
			update_lights(scene, gpu_lights, nr_of_gpu_lights);
			update_rectangular_area_lights(scene, gpu_rect_lights, nr_of_gpu_rect_lights);
			crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));
			set.rt->trace();

			cout << "primary visibility took " << set.basic_rt<B, T>()->timings.front() << "ms." << endl;
	
			cout << "saving output" << endl;
			png::image<png::rgb_pixel> image(w, h);
			vec3f *data = new vec3f[w*h];
// 			cudaMemcpy(data, (dynamic_cast<gpu_material_evaluator<B, T>*>(set.bouncer))->material_colors, sizeof(float3)*w*h, cudaMemcpyDeviceToHost);
			cudaMemcpy(data, (dynamic_cast<gpu_cgls_arealight_evaluator<B, T>*>(set.bouncer))->output_color, sizeof(float3)*w*h, cudaMemcpyDeviceToHost);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					vec3f *pixel = data+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(255*pixel->x,0,255), clamp(255*pixel->y,0,255), clamp(255*pixel->z,0,255))); 
				}
			}
			image.write("out.png");
	}
		
}

/* vim: set foldmethod=marker: */

