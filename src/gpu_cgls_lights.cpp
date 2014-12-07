#include "gpu_cgls_lights.h"

#include "cgls-lights.h"
#include "arealight-sampler.h"
#include "rayvis.h"
#include "util.h"
#include "vars.h"
#include "dofrays.h"
#include "tracers.h"
#include "gpu-pt-kernels.h"	// reset_gpu_buffer

#include <libhyb/trav-util.h>

using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;

extern float aperture, focus_distance;

namespace local {

// 	vec3f *material;
	typedef cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> crgs_with_diffs;

	template<typename _box_t, typename _tri_t> struct gpu_cgls_light_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		cuda::cgls::light *lights;
		int nr_of_lights;
		gpu_cgls_light_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, 
								 crgs_with_diffs *crgs, cuda::cgls::light *lights, int nr_of_lights)
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
									 crgs_with_diffs *crgs, cuda::cgls::rect_light *lights, int nr_of_lights,
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
			rta::cuda::cgls::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
													   this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
													   primary_intersection, this->tri_ptr, uniform_random_numbers, potential_sample_contribution, 
													   pi);
		}
		virtual void integrate_light_sample() {
			rta::cuda::cgls::integrate_light_sample(this->w, this->h, this->gpu_last_intersection, potential_sample_contribution,
													this->material_colors, output_color, curr_bounce-1);
		}
		virtual void bounce() {
			cout << "bounce " << curr_bounce << endl;
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
			}
			++curr_bounce;
		}
		virtual std::string identification() {
			return "evaluate first-hit material and shade with cgls lights.";
		}
	};


	void gpu_cgls_lights_arealight_sampler::activate(rt_set *orig_set) {
		if (activated) return;
		gi_algorithm::activate(orig_set);
		set = *orig_set;
		set.rt = set.rt->copy();
		if (scene.id >= 0) {
			gpu_lights = cuda::cgls::convert_and_upload_lights(scene, nr_of_gpu_lights);
			gpu_rect_lights = cuda::cgls::convert_and_upload_rectangular_area_lights(scene, nr_of_gpu_rect_lights);
		}
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(w, h);
// 		set.bouncer = new gpu_material_evaluator<B, T>(w, h, gpu_materials, triangles, crgs);
// 		set.bouncer = new gpu_cgls_light_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights);
		gi::cuda::halton_pool2f pool = gi::cuda::generate_halton_pool_on_gpu(w*h);
		set.bouncer = new gpu_cgls_arealight_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_rect_lights, nr_of_gpu_rect_lights, pool, 32);
		set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
		set.basic_rt<B, T>()->ray_generator(set.rgen);

		cuda::cgls::init_cuda_image_transfer(result);

		cout << "pool: " << pool.N << endl;
			
		vec3f tng1(1,0,0);
		vec3f tng2(0,1,0);
		vec3f n(0,-1,0);
		cout << "T " << make_tangential(tng1, n) << endl;
		cout << "T " << make_tangential(tng2, n) << endl;
	}

	void gpu_cgls_lights_arealight_sampler::update() {
		if (shadow_tracer && shadow_tracer->progressive_trace_running()) {
			shadow_tracer->trace_progressively(false);
			gpu_cgls_arealight_evaluator<B,T> *bouncer = dynamic_cast<gpu_cgls_arealight_evaluator<B, T>*>(set.bouncer);
			float3 *colors = bouncer->output_color;
// 			float3 *colors = bouncer->material_colors;
			cuda::cgls::copy_cuda_image_to_texture(w, h, colors, 1.0f);
		}
	}

	void gpu_cgls_lights_arealight_sampler::compute() {
			cout << "restarting progressive display" << endl;
			vec3f pos, dir, up;
			matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
			extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
			extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
			extract_up_vec3f_of_matrix(&up, lookat_matrix);
			update_lights(scene, gpu_lights, nr_of_gpu_lights);
			update_rectangular_area_lights(scene, gpu_rect_lights, nr_of_gpu_rect_lights);
			crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

			set.rt->trace_progressively(true);
			shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	
			/*
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
			*/
	}


	// 
	// CGLS LIGHTS, FOR REAL
	//
	void gpu_cgls_lights::activate(rt_set *orig_set) {
		if (activated) return;
		gi_algorithm::activate(orig_set);

		scm_c_eval_string("(if (defined? 'setup-lights) (setup-lights))");

		set = *orig_set;
		set.rt = set.rt->copy();
		gpu_lights = cuda::cgls::convert_and_upload_lights(scene, nr_of_gpu_lights);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(w, h);
		set.bouncer = new gpu_cgls_light_evaluator<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights);
		gi::cuda::halton_pool2f pool = gi::cuda::generate_halton_pool_on_gpu(w*h);
		set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
		set.basic_rt<B, T>()->ray_generator(set.rgen);

		cuda::cgls::init_cuda_image_transfer(result);

		cout << "pool: " << pool.N << endl;
			
		vec3f tng1(1,0,0);
		vec3f tng2(0,1,0);
		vec3f n(0,-1,0);
		cout << "T " << make_tangential(tng1, n) << endl;
		cout << "T " << make_tangential(tng2, n) << endl;
	}

	void gpu_cgls_lights::update() {
		if (shadow_tracer && shadow_tracer->progressive_trace_running()) {
			shadow_tracer->trace_progressively(false);
			gpu_cgls_light_evaluator<B,T> *bouncer = dynamic_cast<gpu_cgls_light_evaluator<B, T>*>(set.bouncer);
// 			float3 *colors = bouncer->output_color;
			float3 *colors = bouncer->material_colors;
			cuda::cgls::copy_cuda_image_to_texture(w, h, colors, 1.0f);
		}
	}

	void gpu_cgls_lights::compute() {
			cout << "restarting progressive display" << endl;
			vec3f pos, dir, up;
			matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
			extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
			extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
			extract_up_vec3f_of_matrix(&up, lookat_matrix);
			update_lights(scene, gpu_lights, nr_of_gpu_lights);
			crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

			set.rt->trace_progressively(true);
			shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	
			gpu_cgls_light_evaluator<B,T> *bouncer = dynamic_cast<gpu_cgls_light_evaluator<B, T>*>(set.bouncer);
			float3 *colors = bouncer->material_colors;
			cuda::cgls::copy_cuda_image_to_texture(w, h, colors, 1.0f);

			/*
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
			*/
	}
	
	// 
	// CGLS LIGHTS, WITH DOF
	//

	static bool alphamaps = true;

	/*! \brief An extension of \ref rta::cuda::cam_ray_generator_shirley that
	 *  computes ray origins on the lens and adapts the directions so that all
	 *  rays generated for a given pixel will converge on the focal plane.
	*/
	class lens_ray_generator : public rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> {
	protected:
		gi::cuda::mt_pool3f jitter;
	public:
		float focus_distance, aperture, eye_to_lens;
		lens_ray_generator(uint res_x, uint res_y, 
						   float focus_distance, float aperture, float eye_to_lens,
						   gi::cuda::mt_pool3f jitter) 
		: rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(res_x, res_y),
		  focus_distance(focus_distance), aperture(aperture), eye_to_lens(eye_to_lens),
		  jitter(jitter) {
		}
		virtual void generate_rays() {
			rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>::generate_rays();
			rta::cuda::setup_shirley_lens_rays(this->gpu_direction, this->gpu_origin, this->gpu_maxt, 
											   fovy, aspect, this->w, this->h, (float3*)&dir, (float3*)&position, (float3*)&up, FLT_MAX,
											   focus_distance, aperture, eye_to_lens, jitter);
		}
		virtual std::string identification() { return "cuda version of the ray generator according to shirley."; }
		virtual void dont_forget_to_initialize_max_t() {}
	};
	
	template<typename _box_t, typename _tri_t> struct gpu_cgls_light_evaluator_dof : public gpu_cgls_light_evaluator<forward_traits> {
		declare_traits_types;
		int w, h;
		int samples, curr;
		float3 *output_color;
		gpu_cgls_light_evaluator_dof(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, 
									 crgs_with_diffs *crgs, cuda::cgls::light *lights, int nr_of_lights, int samples)
		: gpu_cgls_light_evaluator<forward_traits>(w, h, materials, triangles, crgs, lights, nr_of_lights), w(w), h(h), samples(samples), curr(0) {
			checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
		}
		void new_pass() {
			reset_gpu_buffer(output_color, w, h, make_float3(0,0,0));
			curr = 0;
		}
		virtual void bounce() {
			cout << "sample " << curr << " taken" << endl;
			this->evaluate_material();
			this->shade_locally();
			combine_color_samples(output_color, w, h, this->material_colors, curr);
			++curr;
		}
		virtual bool trace_further_bounces() {
			return curr < samples;
		}
		virtual std::string identification() {
			return "repeatedly evaluate first-hit material and shade with cgls lights using the lens_ray_generator.";
		}
	};
	

	void gpu_cgls_lights_dof::activate(rt_set *orig_set) {
		if (activated) return;
		gi_algorithm::activate(orig_set);

		jitter = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		scm_c_eval_string("(if (defined? 'setup-lights) (setup-lights))");

		set = *orig_set;
		set.rgen = crgs = new lens_ray_generator(w, h, focus_distance, aperture, eye_to_lens, jitter);
		if (!alphamaps) {
			set.rt = set.rt->copy();
		}
		else {
			typedef cuda::indexed_binary_bvh<B, T, rta::indexed_binary_bvh<B, T, cuda::bbvh_node_float4<B>>> ibvh_t;
			ibvh_t *bvh = dynamic_cast<ibvh_t*>(set.as);
			if (!bvh)
				throw std::logic_error("we assume that the plugin is loaded with -l 2f4 -A -b bsah -t cis");
			set.rt = new rta::cuda::bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas<B,T,ibvh_t>(set.rgen, bvh, 0, gpu_materials, jitter);
		}
		gpu_lights = cuda::cgls::convert_and_upload_lights(scene, nr_of_gpu_lights);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.bouncer = new gpu_cgls_light_evaluator_dof<B, T>(w, h, gpu_materials, triangles, crgs, gpu_lights, nr_of_gpu_lights, 32);
		set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
		set.basic_rt<B, T>()->ray_generator(set.rgen);

		cuda::cgls::init_cuda_image_transfer(result);
	}
	void gpu_cgls_lights_dof::update() {
		update_mt_pool(jitter);
		if (set.rt->progressive_trace_running()) {
			update_mt_pool(jitter);
			set.rgen->generate_rays();
			set.rt->trace_progressively(false);
			gpu_cgls_light_evaluator_dof<B,T> *bouncer = dynamic_cast<gpu_cgls_light_evaluator_dof<B, T>*>(set.bouncer);
// 			float3 *shaded = bouncer->material_colors;
			float3 *accum = bouncer->output_color;
			cuda::cgls::copy_cuda_image_to_texture(w, h, accum, 1.0f);
		}
	}
	void gpu_cgls_lights_dof::compute() {
		update_mt_pool(jitter);
		gpu_cgls_lights::compute();
		if (aperture != ::aperture || focus_distance != ::focus_distance) {
			aperture = ::aperture;
			focus_distance = ::focus_distance;
			lens_ray_generator *rg = dynamic_cast<lens_ray_generator*>(crgs);
			rg->aperture = aperture;
			rg->focus_distance = focus_distance;
		}
	}

}

/* vim: set foldmethod=marker: */

