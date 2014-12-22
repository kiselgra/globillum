#ifndef __GI_GPU_PT_H__ 
#define __GI_GPU_PT_H__ 

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"	// contains gpu_material_evaluator.
#include "arealight-sampler.h"
#include "gpu-pt-kernels.h"

#include "rayvis.h"
#include "vars.h"
#include "util.h"

#include <iostream>

template<typename _box_t, typename _tri_t> struct tandem_tracer : public rta::raytracer {
	declare_traits_types;
	
	typedef rta::basic_raytracer<box_t, tri_t> tracer_t;
	tracer_t *closest_hit_tracer;
	tracer_t *any_hit_tracer;
	tracer_t *use_tracer, *other, *last;

	tandem_tracer(tracer_t *ch, tracer_t *ah) : closest_hit_tracer(ch), any_hit_tracer(ah), use_tracer(0), other(0), last(0) {
	}
	void select_closest_hit_tracer() {
		use_tracer = closest_hit_tracer;
		other = any_hit_tracer;
	}
	void select_any_hit_tracer() {
		use_tracer = any_hit_tracer;
		other = closest_hit_tracer;
	}
	virtual void trace() {
		throw std::logic_error("a tandem tracer is for progressive tracing, only.");
	}
	// bounce() might change the tracer, or might keep it
	// therefore, after bounce() and tracer_furhter_boucnes() was called with the `last'
	// tracer, we copy its information over to both versions.
	virtual void trace_progressively(bool first) {
		last = use_tracer;
		use_tracer->trace_progressively(first);
		use_tracer->copy_progressive_state(last);
		other->copy_progressive_state(last);
	}
	virtual std::string identification() {
		return std::string("wrapper to trace using two basic_raytracers in tandem, in this case: (")
			   + closest_hit_tracer->identification() + ", and " + any_hit_tracer->identification() + ")";
	}
	virtual bool progressive_trace_running() {
		return use_tracer->progressive_trace_running();
	}
	virtual rta::raytracer* copy() {
		return new tandem_tracer(*this);
	}
	virtual bool supports_max_t() {
		return closest_hit_tracer->supports_max_t() && any_hit_tracer->supports_max_t();
	}
};

struct light_sample_ray_storage : public rta::ray_generator, rta::cuda::gpu_ray_generator {
	light_sample_ray_storage(uint w, uint h) : rta::ray_generator(w, h), rta::cuda::gpu_ray_generator(w, h) {
	}
	virtual std::string identification() {
		return std::string("just a storage unit for the light sample rays, and a means to bind them to the appropriate tracer.");
	}
	virtual void generate_rays() {
		throw std::logic_error("light_sample_ray_storage is just ray storage filled in by a bouncer. it can't generate rays itself.");
	}
	virtual void dont_forget_to_initialize_max_t() {}
};

template<typename _box_t, typename _tri_t> struct gpu_pt_bouncer : public rta::cuda::gpu_ray_bouncer<forward_traits> {
	declare_traits_types;
	rta::cuda::material_t *materials;
	rta::cuda::simple_triangle *tri_ptr;
	rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
	gi::light *lights;
	int nr_of_lights;
	enum random_number_generator_t { none, simple_halton, lcg, per_frame_mt };
	random_number_generator_t rnd_type;
	gi::halton_pool2f rnd_halton;
	gi::lcg_random_state rnd_lcg;
	gi::cuda::mt_pool3f rnd_mt_light, rnd_mt_path;
	uint w, h;
	int curr_path;
	int curr_bounce;
	int path_len;
	int max_path_len;
	int path_samples;
	rta::triangle_intersection<rta::cuda::simple_triangle> *path_intersections,
	                                                       *shadow_intersections;
	float3 *output_color, *path_accum_color, *throughput, *potential_sample_contribution;
	float *light_sample_directions, *path_sample_directions;
	float *light_sample_origins, *path_sample_origins;
	float *light_sample_maxt, *path_sample_maxt;
	
	// maintain which tracer to use for the next bounce
	tandem_tracer<box_t, tri_t> *tracers;
	void register_tracers(tandem_tracer<box_t, tri_t> *tt) {
		tracers = tt;
		light_sample_ray_storage *gpurg = new light_sample_ray_storage(w, h);
		tracers->any_hit_tracer->ray_generator(gpurg);
		light_sample_directions = gpurg->gpu_direction;
		light_sample_origins = gpurg->gpu_origin;
		light_sample_maxt = gpurg->gpu_maxt;
	}

	gpu_pt_bouncer(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles,
				   rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs, 
				   gi::light *lights, int nr_of_lights, int max_path_len, int path_samples)
	: rta::cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), tri_ptr(triangles), crgs(crgs),
	  lights(lights), nr_of_lights(nr_of_lights), rnd_type(none), w(w), h(h),
	  curr_bounce(0), path_len(0), max_path_len(max_path_len), curr_path(0), path_samples(path_samples), output_color(0), tracers(0),
	  light_sample_origins(0), light_sample_directions(0), light_sample_maxt(0),
	  path_sample_origins(0), path_sample_directions(0), path_sample_maxt(0)
	{
		checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&path_accum_color, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&throughput, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&potential_sample_contribution, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&shadow_intersections, sizeof(rta::triangle_intersection<rta::cuda::simple_triangle>)*w*h));
		path_intersections = this->gpu_last_intersection;
		// path sample directions *have* to be the crgs directions as the brdf requires valid 'last direction' in the data.
		path_sample_directions = crgs->gpu_direction;	
		path_sample_origins = crgs->gpu_origin;
		path_sample_maxt = crgs->gpu_maxt;
	}
	~gpu_pt_bouncer() {
		checked_cuda(cudaFree(output_color));
		checked_cuda(cudaFree(path_accum_color));
		checked_cuda(cudaFree(shadow_intersections));
	}
	virtual void random_number_generator(gi::halton_pool2f rng) {
		rnd_halton = rng;
		rnd_type = simple_halton;
	}
	virtual void random_number_generator(gi::lcg_random_state rng) {
		rnd_lcg = rng;
		rnd_type = lcg;
	}
	virtual void random_number_generator(gi::cuda::mt_pool3f rng_light, gi::cuda::mt_pool3f rng_path) {
		rnd_mt_light = rng_light;
		rnd_mt_path = rng_path;
		rnd_type = per_frame_mt;
	}
	virtual void new_pass() {
		curr_bounce = curr_path = 0;
		path_len = 0;
		gi::cuda::reset_gpu_buffer(throughput, w, h, make_float3(1,1,1));
		gi::cuda::reset_gpu_buffer(path_accum_color, w, h, make_float3(0,0,0));
		gi::cuda::reset_gpu_buffer(output_color, w, h, make_float3(0,0,0));
		restart_rayvis();
	}
	virtual void new_path() {
		path_len = 0;
		this->gpu_last_intersection = path_intersections;
		gi::cuda::combine_color_samples(output_color, w, h, path_accum_color, curr_path);
		gi::cuda::reset_gpu_buffer(throughput, w, h, make_float3(1,1,1));
		gi::cuda::reset_gpu_buffer(path_accum_color, w, h, make_float3(0,0,0));
		this->crgs->generate_rays();
		curr_path++;
	}
	virtual void setup_new_arealight_sample() {
		gi::cuda::random_sampler_path_info pi;
		pi.curr_path = curr_path;
		pi.curr_bounce = path_len;
		pi.max_paths = path_samples;
		pi.max_bounces = max_path_len;
		if (rnd_type == simple_halton)
			rta::cuda::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
												 light_sample_origins, light_sample_directions, light_sample_maxt,
												 path_intersections, this->tri_ptr, rnd_halton, potential_sample_contribution, 
												 pi);
		else if (rnd_type == lcg)
			rta::cuda::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
												 light_sample_origins, light_sample_directions, light_sample_maxt,
												 path_intersections, this->tri_ptr, rnd_lcg, potential_sample_contribution, 
												 pi);
		else if (rnd_type == per_frame_mt)
			rta::cuda::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
												 light_sample_origins, light_sample_directions, light_sample_maxt,
												 path_intersections, this->tri_ptr, rnd_mt_light, potential_sample_contribution, 
												 pi);
		else throw std::logic_error("unsupported random number generator in setup_new_arealight_sample!");
	}
	virtual void compute_path_contrib_and_bounce() {
		gi::cuda::random_sampler_path_info pi;
		pi.curr_path = curr_path;
		pi.curr_bounce = path_len;
		pi.max_paths = path_samples;
		pi.max_bounces = max_path_len;
		if (rnd_type == simple_halton)
			compute_path_contribution_and_bounce(this->w, this->h, path_sample_origins, path_sample_directions, path_sample_maxt,
												 this->crgs->differentials_origin, this->crgs->differentials_direction,
												 path_intersections, this->tri_ptr, this->materials, rnd_halton, throughput, path_accum_color,
												 light_sample_directions, shadow_intersections, potential_sample_contribution,
												 pi);
		else if (rnd_type == lcg)
			compute_path_contribution_and_bounce(this->w, this->h, path_sample_origins, path_sample_directions, path_sample_maxt,
												 this->crgs->differentials_origin, this->crgs->differentials_direction,
												 path_intersections, this->tri_ptr, this->materials, rnd_lcg, throughput, path_accum_color,
												 light_sample_directions, shadow_intersections, potential_sample_contribution,
												 pi);
		else if (rnd_type == per_frame_mt)
			compute_path_contribution_and_bounce(this->w, this->h, path_sample_origins, path_sample_directions, path_sample_maxt,
												 this->crgs->differentials_origin, this->crgs->differentials_direction,
												 path_intersections, this->tri_ptr, this->materials, rnd_mt_path, throughput, path_accum_color,
												 light_sample_directions, shadow_intersections, potential_sample_contribution,
												 pi);
		else throw std::logic_error("unsupported random number generator in setup_new_arealight_sample!");
	}
	virtual void bounce() {
		bool compute_light_sample = false;
		bool compute_path_segment = false;
			
		std::cout << "bounce " << curr_bounce << " (path " << curr_path << ")" << std::endl;

		if (path_len == 0 && this->gpu_last_intersection == path_intersections) {
			vec3f campos = this->crgs->position;
			std::cout << " - adding camera position as vertex" << std::endl;
			add_vertex_to_all_rays(make_float3(campos.x, campos.y, campos.z));
			std::cout << " - adding primary hit as vertex" << std::endl;
			add_intersections_to_rays(this->w, this->h, this->gpu_last_intersection, this->tri_ptr);
			compute_light_sample = true;
		}
		else {
			if (this->gpu_last_intersection == shadow_intersections) {
				compute_path_segment = true;
				std::cout << "shadow trace: " << tracers->any_hit_tracer->timings[path_len] << std::endl;
			}
			else {
				std::cout << " - add path intersection as vertex" << std::endl;
				add_intersections_to_rays(this->w, this->h, path_intersections, this->tri_ptr);
				compute_light_sample = true;
			}
		}

		if (compute_light_sample) {
			std::cout << " - computing area light sample" << std::endl;
			setup_new_arealight_sample();
			this->gpu_last_intersection = shadow_intersections;
			tracers->select_any_hit_tracer();
		}
		if (compute_path_segment) {
			std::cout << " - computing new path sample" << std::endl;
			compute_path_contrib_and_bounce();
			this->gpu_last_intersection = path_intersections;
			tracers->select_closest_hit_tracer();
			// a light has been sampled, so the current path is finished.
			++path_len;
			if (rnd_type == per_frame_mt) {
				update_mt_pool(rnd_mt_light);
				update_mt_pool(rnd_mt_path);
			}
		}

		++curr_bounce;
	}
	virtual bool trace_further_bounces() {
// 		std::cout<<"cb: " << curr_bounce << std::endl;
		if (path_len < max_path_len)
			return true;
		else
			new_path();
		if (curr_path < path_samples) {
			return true;
		}
		return false;
// 		return curr_bounce < 4;
	}
	virtual std::string identification() {
		return "gpu path tracer";
	}
};

class gpu_pt : public gi_algorithm {
	typedef rta::cuda::simple_aabb B;
	typedef rta::cuda::simple_triangle T;
protected:
	int w, h;
	rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
	rta::rt_set set;
	scene_ref scene;
	gi::light *gpu_rect_lights;
	int nr_of_gpu_rect_lights;
	gpu_pt_bouncer<B, T> *pt;
	tandem_tracer<B, T> *tracer;
	rta::raytracer *shadow_tracer;
public:
	gpu_pt(int w, int h, scene_ref scene, const std::string &name = "gpu_pt")
		: gi_algorithm(name), w(w), h(h),
		  crgs(0), scene(scene), gpu_rect_lights(0), shadow_tracer(0) {
	}

	virtual void activate(rta::rt_set *orig_set);
	virtual void compute();
	virtual void update();
	virtual bool progressive() { return true; }
};



#endif

