#include "direct-lighting.h"

#include "cgls-lights.h"
#include "arealight-sampler.h"
#include "rayvis.h"
#include "util.h"
#include "vars.h"
#include "raygen.h"
#include "tracers.h"

#include <libhyb/trav-util.h>

using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;
extern cuda::material_t *cpu_materials;

extern float aperture, focus_distance, eye_to_lens;

namespace local {

	typedef cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> crgs_with_diffs;

	// 
	// gpu area light evaluator
	// 

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
			rta::cuda::generate_arealight_sample(this->w, this->h, lights, nr_of_lights, overall_light_power,
												 this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
												 primary_intersection, this->tri_ptr, uniform_random_numbers, potential_sample_contribution, 
												 pi);
		}
		virtual void integrate_light_sample() {
			rta::cuda::integrate_light_sample(this->w, this->h, this->gpu_last_intersection, potential_sample_contribution,
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


	// 
	// gpu area light sampler (the `bouncer')
	// 

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
		if (original_subd_set) {
			subd_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(original_subd_set->rt);
			subd_tracer->ray_generator(set.rgen);
			subd_tracer->ray_bouncer(set.bouncer);
			cout << "SUBD TR " << subd_tracer << endl;
			tracers->append_tracer(subd_tracer);
		}
// 		tracers->append_tracer(gpu_tracer);

		if (scene.id >= 0)
			cuda::cgls::init_cuda_image_transfer(result);
	}

	void gpu_arealight_sampler::update() {
		if (shadow_tracers && shadow_tracers->progressive_trace_running()) {
			shadow_tracers->trace_progressively(false);
			gpu_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_arealight_evaluator<B, T>*>(set.bouncer);
			float3 *colors = bouncer->output_color;
// 			float3 *colors = bouncer->material_colors;
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
		if (original_subd_set) {
			rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>
				*subd_shadow_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(subd_tracer->matching_any_hit_tracer());
			shadow_tracers->append_tracer(subd_shadow_tracer);
		}
	}

	void gpu_arealight_sampler::light_samples(int n) {
		cout << name << " is accumulating " << n << " direct lighting samples, now." << endl;
		gpu_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->samples = n;
	}


	// 
	// gpu/cpu area light sampler (the `bouncer')
	// 

	void hybrid_arealight_sampler::activate(rta::rt_set *orig_set) {
		if (activated) return;
		gi_algorithm::activate(orig_set);
		set = *orig_set;
		set.rt = set.rt->copy();
		rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer> 
			*gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(set.rt);
		nr_of_lights = gi::lights.size();
		cpu_lights = new gi::light[gi::lights.size()];	//gi::cuda::convert_and_upload_lights(nr_of_lights, overall_light_power);
		overall_light_power = 0;
		for (int i = 0; i < gi::lights.size(); ++i) {
			cpu_lights[i] = gi::lights[i];
			overall_light_power += cpu_lights[i].power;
		}
		triangles = set.basic_as<B, T>()->canonical_triangle_ptr();
		gi::cuda::mt_pool3f jitter = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		update_mt_pool(jitter);
		set.rgen = crgs = new rta::cuda::jittered_lens_ray_generator(w, h, focus_distance, aperture, eye_to_lens, jitter);
// 		set.rgen = crgs = new rta::cuda::jittered_ray_generator(w, h, jitter);
// 		set.rgen = crgs = new cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(w, h);
		gi::cuda::mt_pool3f pool = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		update_mt_pool(pool);
		
		int light_samples = init_light_samples;
		int path_samples = init_path_samples;
		// shadow rays: light_samples x path_samples.
		set.bouncer = new hybrid_arealight_evaluator<B, T>(w, h, cpu_materials, triangles, crgs, cpu_lights, nr_of_lights, pool, jitter, light_samples, path_samples);
		gpu_tracer->ray_bouncer(set.bouncer);
		gpu_tracer->ray_generator(set.rgen);

		tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::closest_hit_tracer>(gpu_tracer);
		if (original_subd_set) {
			subd_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(original_subd_set->rt);
			subd_tracer->ray_generator(set.rgen);
			subd_tracer->ray_bouncer(set.bouncer);
			cout << "SUBD TR " << subd_tracer << endl;
			tracers->append_tracer(subd_tracer);
		}

		if (scene.id >= 0)
			cuda::cgls::init_cuda_image_transfer(result);
	}

	bool hybrid_arealight_sampler::in_progress() {
		bool cam_sample_running = (shadow_tracer && shadow_tracer->progressive_trace_running());
		if (cam_sample_running) return true;
		hybrid_arealight_evaluator<B,T> *bouncer = dynamic_cast<hybrid_arealight_evaluator<B, T>*>(set.bouncer);
		return (bouncer->curr_cam_sample < bouncer->cam_samples);
	}

	void hybrid_arealight_sampler::update() {
		if (shadow_tracers)  {
			hybrid_arealight_evaluator<B,T> *bouncer = dynamic_cast<hybrid_arealight_evaluator<B, T>*>(set.bouncer);
			if (shadow_tracers->progressive_trace_running()) {
				shadow_tracers->trace_progressively(false);
				float3 *colors = bouncer->output_color.data;
				gi::save_image("arealightsampler", bouncer->image_nr, w, h, colors);
			}
			else {
				if (bouncer->curr_cam_sample < bouncer->cam_samples) {
// 					set.rt->trace_progressively(true);
					tracers->trace_progressively(true);
					dynamic_cast<basic_raytracer<B,T>*>(shadow_tracers)->copy_progressive_state(tracers);
					float3 *colors = bouncer->material_colors.data;
					// gi::save_image("arealightsampler", bouncer->curr_cam_sample * bouncer->light_samples_per_cam_sample + bouncer->curr_light_sample, w, h, colors);
				}
			}
		}
	}

	void hybrid_arealight_sampler::compute() {
		cout << "restarting progressive display" << endl;
		vec3f pos, dir, up;
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
		extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
		extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
		extract_up_vec3f_of_matrix(&up, lookat_matrix);
		if (nr_of_lights != gi::lights.size())
			throw std::runtime_error("number of lights changed in " "hybrid_arealight_sampler::compute" 
									 " this exception is just for consistency to the gpu version.");
		for (int i = 0; i < gi::lights.size(); ++i)
			cpu_lights[i] = gi::lights[i];
		hybrid_arealight_evaluator<B,T> *bouncer = dynamic_cast<hybrid_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->reset_samples();
		bouncer->overall_light_power = overall_light_power;
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

// 		set.rt->trace_progressively(true);
		tracers->trace_progressively(true);

		float3 *colors = bouncer->material_colors.data;
		gi::save_image("arealightsampler", 0, w, h, colors);

		shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
		rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer> 
			*shadow_gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(shadow_tracer);
		
		shadow_tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::any_hit_tracer>(shadow_gpu_tracer);
		shadow_tracers->copy_progressive_state(tracers);

		if (original_subd_set) {
			rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>
				*subd_shadow_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(subd_tracer->matching_any_hit_tracer());
			shadow_tracers->append_tracer(subd_shadow_tracer);
		}
	}

	void hybrid_arealight_sampler::light_samples(int n) {
		cout << name << " is accumulating " << n << " direct lighting samples per camera sample, now." << endl;
		hybrid_arealight_evaluator<B,T> *bouncer = dynamic_cast<hybrid_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->light_samples_per_cam_sample = n;
	}

	void hybrid_arealight_sampler::path_samples(int n) {
		cout << name << " is accumulating " << n << " direct lighting camera samples, now." << endl;
		hybrid_arealight_evaluator<B,T> *bouncer = dynamic_cast<hybrid_arealight_evaluator<B, T>*>(set.bouncer);
		bouncer->cam_samples = n;
	}



}


/* vim: set foldmethod=marker: */

