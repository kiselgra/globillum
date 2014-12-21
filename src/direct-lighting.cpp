#include "direct-lighting.h"

#include "cgls-lights.h"
#include "arealight-sampler.h"
#include "rayvis.h"
#include "util.h"
#include "vars.h"
#include "dofrays.h"
#include "tracers.h"

#include <libhyb/trav-util.h>

using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;

extern float aperture, focus_distance;

namespace local {

	typedef cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> crgs_with_diffs;

	template<typename _box_t, typename _tri_t> struct gpu_arealight_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		gi::light *lights;
		int nr_of_lights;
		gi::cuda::mt_pool3f uniform_random_numbers;
		float3 *potential_sample_contribution; 
		int curr_bounce;
		int samples;
		float3 *output_color;
		float overall_light_power;
		triangle_intersection<cuda::simple_triangle> *primary_intersection;
		gpu_arealight_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, 
									 crgs_with_diffs *crgs, gi::light *lights, int nr_of_lights,
									 gi::cuda::mt_pool3f rnd, int samples)
		: gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs), 
		  lights(lights), nr_of_lights(nr_of_lights), uniform_random_numbers(rnd), samples(samples) {
			  checked_cuda(cudaMalloc(&potential_sample_contribution, sizeof(float3)*w*h));
			  checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
			  checked_cuda(cudaMalloc(&primary_intersection, sizeof(triangle_intersection<cuda::simple_triangle>)*w*h));
			  curr_bounce = 0;
			  this->background = make_float3(1,1,1);
		}
		~gpu_arealight_evaluator() {
			checked_cuda(cudaFree(potential_sample_contribution));
			checked_cuda(cudaFree(output_color));
			checked_cuda(cudaFree(primary_intersection));
		}
		virtual void new_pass() {
			curr_bounce = 0;
			restart_rayvis();
		}
		virtual bool trace_further_bounces() {
			return (curr_bounce < samples+1);
		}
		virtual void setup_new_arealight_sample() {
			gi::cuda::random_sampler_path_info pi;
			pi.curr_path = 0;
			pi.curr_bounce = curr_bounce;
			pi.max_paths = 1;
			pi.max_bounces = samples;
			rta::cuda::cgls::generate_arealight_sample(this->w, this->h, lights, nr_of_lights, overall_light_power,
													   this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
													   primary_intersection, this->tri_ptr, uniform_random_numbers, potential_sample_contribution, 
													   pi);
		}
		virtual void integrate_light_sample() {
			rta::cuda::cgls::integrate_light_sample(this->w, this->h, this->gpu_last_intersection, potential_sample_contribution,
													this->material_colors, output_color, curr_bounce-1);
		}
		virtual void bounce() {
			cout << "direct lighting sample " << curr_bounce << endl;
			if (curr_bounce == 0) {
				// this has to be done before switching the intersection data.
				this->evaluate_material();
				vec3f campos = this->crgs->position;
				add_vertex_to_all_rays(make_float3(campos.x, campos.y, campos.z));
				add_intersections_to_rays(this->w, this->h, this->gpu_last_intersection, this->tri_ptr);
				triangle_intersection<cuda::simple_triangle> *tmp = primary_intersection;
				primary_intersection = this->gpu_last_intersection;
				this->gpu_last_intersection = tmp;
				setup_new_arealight_sample();
			}
			else if (curr_bounce > 0) {
				if (curr_bounce < 3)
					add_intersections_to_rays(this->w, this->h, this->gpu_last_intersection, this->tri_ptr);
				integrate_light_sample();
				if (curr_bounce < samples)
					setup_new_arealight_sample();
				update_mt_pool(uniform_random_numbers);
			}
			++curr_bounce;
		}
		virtual std::string identification() {
			return "evaluate first-hit material and shade with cgls lights.";
		}
	};


	void gpu_arealight_sampler::activate(rt_set *orig_set) {
		if (activated) return;
		gi_algorithm::activate(orig_set);
		set = *orig_set;
		set.rt = set.rt->copy();
		rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer> 
			*gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(set.rt);
		gpu_lights = gi::cuda::convert_and_upload_lights(nr_of_gpu_lights, overall_light_power);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(w, h);
		gi::cuda::mt_pool3f pool = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		
		set.bouncer = new gpu_arealight_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights, pool, 32);
		gpu_tracer->ray_bouncer(set.bouncer);
		gpu_tracer->ray_generator(set.rgen);
		tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::closest_hit_tracer>(gpu_tracer);
		tracers->append_tracer(gpu_tracer);
// 		tracers->append_tracer(gpu_tracer);

		if (scene.id >= 0)
			cuda::cgls::init_cuda_image_transfer(result);
	}

	void gpu_arealight_sampler::update() {
		if (shadow_tracers && shadow_tracers->progressive_trace_running()) {
			shadow_tracers->trace_progressively(false);
			gpu_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_arealight_evaluator<B, T>*>(set.bouncer);
			float3 *colors = bouncer->output_color;
			if (scene.id >= 0)
				cuda::cgls::copy_cuda_image_to_texture(w, h, colors, 1.0f);
			else
				gi::cuda::download_and_save_image("arealightsampler", bouncer->curr_bounce, w, h, colors);

		}
	}
	bool gpu_arealight_sampler::in_progress() {
		return (shadow_tracers && shadow_tracers->progressive_trace_running());
	}

	void gpu_arealight_sampler::compute() {
		cout << "restarting progressive display" << endl;
		vec3f pos, dir, up;
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
		extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
		extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
		extract_up_vec3f_of_matrix(&up, lookat_matrix);
		gi::cuda::update_lights(gpu_lights, nr_of_gpu_lights, overall_light_power);
		gpu_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->overall_light_power = overall_light_power;
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

		tracers->trace_progressively(true);
		shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
		rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer> 
			*shadow_gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(shadow_tracer);
		
		float3 *colors = bouncer->material_colors;
		gi::cuda::download_and_save_image("arealightsampler", 0, w, h, colors);

		shadow_tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::any_hit_tracer>(shadow_gpu_tracer);
		shadow_tracers->copy_progressive_state(tracers);
		shadow_tracers->append_tracer(shadow_gpu_tracer);
// 		shadow_tracers->append_tracer(shadow_gpu_tracer);
	}

	void gpu_arealight_sampler::light_samples(int n) {
		cout << name << " is accumulating " << n << " direct lighting samples, now." << endl;
		gpu_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->samples = n;
	}
}


/* vim: set foldmethod=marker: */

