#ifndef __GI_CPU_PT_H__ 
#define __GI_CPU_PT_H__ 

#include "gpu-pt.h"
#include "tracers.h"	// for tandem_tracer
#include "lights.h"	// for tandem_tracer

void compute_path_contribution_and_bounce(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t, float3 *ray_diff_org, float3 *ray_diff_dir,
										  rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, float3 *uniform_random, float3 *throughput, float3 *col_accum,
										  float3 *to_light, rta::triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::light *skylight, int path_len);
										  

template<typename _box_t, typename _tri_t> struct cpu_pt_bouncer : public rta::cpu_ray_bouncer<forward_traits> {
	declare_traits_types;
	rta::cuda::material_t *materials;
	rta::simple_triangle *tri_ptr;
	// todo: `advanced' ray generators are not avaible as cpu-only versions, atm.
	rta::camera_ray_generator_shirley<rta::ray_generator_with_differentials> *crgs;
	gi::light *lights;
	int nr_of_lights;
	float overall_light_power;
	gi::cuda::mt_pool3f rnd_mt_light, rnd_mt_path;
	float3 *mt_numbers_light, *mt_numbers_path;
	uint w, h;
	int curr_path;
	int curr_bounce;
	int path_len;
	int max_path_len;
	int path_samples;
	rta::triangle_intersection<rta::simple_triangle> *path_intersections,
	                                                 *shadow_intersections;
	float3 *output_color, *path_accum_color, *throughput, *potential_sample_contribution;
	float *gpu_path_sample_directions;
	float *gpu_path_sample_origins;
	float *gpu_path_sample_maxt;
	float3 *light_sample_directions, *path_sample_directions, *path_differentials_directions;
	float3 *light_sample_origins,    *path_sample_origins,    *path_differentials_origins;
	float  *light_sample_maxt,       *path_sample_maxt;
	bool verbose;
	gi::light *skylight;

	// maintain which tracer to use for the next bounce
	rta::tandem_tracer<box_t, tri_t> *tracers;
	void register_tracers(rta::tandem_tracer<box_t, tri_t> *tt) {
		// generate new ray generator for the 
		tracers = tt;
	}

	cpu_pt_bouncer(uint w, uint h, rta::cuda::material_t *materials, rta::simple_triangle *triangles,
				   rta::camera_ray_generator_shirley<rta::ray_generator_with_differentials> *crgs, 
				   gi::light *lights, int nr_of_lights, int max_path_len, int path_samples)
	: rta::cpu_ray_bouncer<forward_traits>(w, h),
	  materials(materials), tri_ptr(triangles), crgs(crgs),
	  lights(lights), nr_of_lights(nr_of_lights), w(w), h(h),
	  curr_bounce(0), path_len(0), max_path_len(max_path_len), curr_path(0), path_samples(path_samples), output_color(0), tracers(0),
	  gpu_path_sample_origins(0), gpu_path_sample_directions(0), gpu_path_sample_maxt(0), verbose(false),
	  skylight(0)
	{
		output_color = new float3[w*h];
		path_accum_color = new float3[w*h];
		throughput = new float3[w*h];
		potential_sample_contribution = new float3[w*h];
		path_intersections = new rta::triangle_intersection<rta::simple_triangle>[w*h];
		shadow_intersections = new rta::triangle_intersection<rta::simple_triangle>[w*h];
		light_sample_directions = new float3[w*h];
		light_sample_origins = new float3[w*h];
		light_sample_maxt = new float[w*h];
		path_sample_directions = new float3[w*h];
		path_sample_origins = new float3[w*h];
		path_sample_maxt = new float[w*h];
		path_differentials_directions = new float3[2*w*h];
		path_differentials_origins = new float3[2*w*h];
		mt_numbers_light = new float3[w*h];
		mt_numbers_path  = new float3[w*h];
		for (gi::light &l : gi::lights) {
			if (l.type == gi::light::sky) {
				skylight = &l;
				break;
			}
		}
	}
	~cpu_pt_bouncer() {
		delete [] output_color;
		delete [] path_accum_color;
		delete [] throughput;
		delete [] potential_sample_contribution;
	}
	virtual void random_number_generator(gi::cuda::mt_pool3f rng_light, gi::cuda::mt_pool3f rng_path) {
		rnd_mt_light = rng_light;
		rnd_mt_path = rng_path;
		checked_cuda(cudaMemcpy(mt_numbers_light, rnd_mt_light.data, sizeof(float3)*this->w*this->h, cudaMemcpyDeviceToHost));
		checked_cuda(cudaMemcpy(mt_numbers_path,  rnd_mt_path.data,  sizeof(float3)*this->w*this->h, cudaMemcpyDeviceToHost));
	}
	virtual void new_pass() {
		curr_bounce = curr_path = 0;
		path_len = 0;
		gi::clear_array(throughput, w, h, make_float3(1,1,1));
		gi::clear_array(path_accum_color, w, h, make_float3(0,0,0));
		gi::clear_array(output_color, w, h, make_float3(0,0,0));
		restart_rayvis();
	}
	virtual void new_path() {
		path_len = 0;
		this->last_intersection.data = path_intersections;
		gi::combine_color_samples(output_color, w, h, path_accum_color, curr_path);
		gi::clear_array(throughput, w, h, make_float3(1,1,1));
		gi::clear_array(path_accum_color, w, h, make_float3(0,0,0));
		this->crgs->generate_rays();
		int w = this->w, h = this->h;
		checked_cuda(cudaMemcpy(path_sample_directions,        gpu_path_sample_directions,          w*h*3*sizeof(float), cudaMemcpyDeviceToHost));
		checked_cuda(cudaMemcpy(path_sample_origins,           gpu_path_sample_origins,             w*h*3*sizeof(float), cudaMemcpyDeviceToHost));
		checked_cuda(cudaMemcpy(path_sample_maxt,              gpu_path_sample_maxt,                w*h*1*sizeof(float), cudaMemcpyDeviceToHost));
		checked_cuda(cudaMemcpy(path_differentials_directions, this->crgs->differentials_direction.data, w*h*6*sizeof(float), cudaMemcpyDeviceToHost));
		checked_cuda(cudaMemcpy(path_differentials_origins,    this->crgs->differentials_origin.data,    w*h*6*sizeof(float), cudaMemcpyDeviceToHost));
		curr_path++;
	}
	virtual void setup_new_arealight_sample() {
		gi::cuda::random_sampler_path_info pi;
		pi.curr_path = curr_path;
		pi.curr_bounce = path_len;
		pi.max_paths = path_samples;
		pi.max_bounces = max_path_len;
		rta::generate_arealight_sample(this->w, this->h, lights, nr_of_lights, overall_light_power,
									   light_sample_origins, light_sample_directions, light_sample_maxt,
									   (rta::triangle_intersection<rta::cuda::simple_triangle>*)path_intersections, 
									   (rta::cuda::simple_triangle*)this->tri_ptr, mt_numbers_light, potential_sample_contribution);
	}
	virtual void compute_path_contrib_and_bounce() {
		compute_path_contribution_and_bounce(this->w, this->h, path_sample_origins, path_sample_directions, path_sample_maxt,
											 path_differentials_origins, path_differentials_directions,
											 (rta::triangle_intersection<rta::cuda::simple_triangle>*)path_intersections, 
											 (rta::cuda::simple_triangle*)this->tri_ptr, this->materials, mt_numbers_path, 
											 throughput, path_accum_color, light_sample_directions, 
											 (rta::triangle_intersection<rta::cuda::simple_triangle>*)shadow_intersections, 
											 potential_sample_contribution, skylight, curr_path);
		if (this->verbose)
			gi::save_image("accum", curr_bounce, w, h, path_accum_color);
	}
	virtual void bounce() {
		bool compute_light_sample = false;
		bool compute_path_segment = false;
			
		std::cout << "bounce " << curr_bounce << " (path " << curr_path << ")" << std::endl;

		if (path_len == 0 && this->last_intersection.data == path_intersections) {
			compute_light_sample = true;
		}
		else {
			if (this->last_intersection.data == shadow_intersections) {
				compute_path_segment = true;
				std::cout << " - (shadow trace: " << tracers->any_hit_tracer->timings[path_len] << ")" << std::endl;
			}
			else {
				compute_light_sample = true;
			}
		}
			
		if (compute_light_sample) {
			std::cout << " - computing area light sample" << std::endl;

			// compute area light sample on cpu.
			setup_new_arealight_sample();
			this->last_intersection.data = shadow_intersections;
			tracers->select_any_hit_tracer();
			
		}
		if (compute_path_segment) {
			std::cout << " - computing new path sample" << std::endl;

			// compute actual bounce
			compute_path_contrib_and_bounce();
			this->last_intersection.data = path_intersections;
			tracers->select_closest_hit_tracer();
		
			// a light has been sampled, so the current sub-path is finished.
			++path_len;
			update_mt_pool(rnd_mt_light);
			update_mt_pool(rnd_mt_path);
			checked_cuda(cudaMemcpy(mt_numbers_light, rnd_mt_light.data, sizeof(float3)*this->w*this->h, cudaMemcpyDeviceToHost));
			checked_cuda(cudaMemcpy(mt_numbers_path,  rnd_mt_path.data,  sizeof(float3)*this->w*this->h, cudaMemcpyDeviceToHost));
		}

		++curr_bounce;
	}
	virtual bool trace_further_bounces() {
		if (path_len < max_path_len)
			return true;
		else
			new_path();
		if (curr_path < path_samples) {
			return true;
		}
		return false;
	}
	virtual std::string identification() {
		return "gpu path tracer";
	}
};

class cpu_pt : public gi_algorithm {
	typedef rta::simple_aabb B;
	typedef rta::simple_triangle T;
protected:
	int w, h;
	rta::camera_ray_generator_shirley<rta::ray_generator_with_differentials> *crgs;
	rta::rt_set set;
	scene_ref scene;
	gi::light *cpu_lights;
	int nr_of_lights;
	cpu_pt_bouncer<B, T> *pt;
	rta::tandem_tracer<B, T> *tracer;
	rta::raytracer *shadow_tracer;
	rta::simple_triangle *triangles;
	float overall_light_power;
	gi::cuda::mt_pool3f jitter; 
	float3 *cpu_jitter;
	rta::iterated_cpu_tracers<B, T, rta::closest_hit_tracer> *tracers;
	rta::iterated_cpu_tracers<B, T, rta::any_hit_tracer> *shadow_tracers;
public:
	cpu_pt(int w, int h, scene_ref scene, const std::string &name = "cpu_pt")
		: gi_algorithm(name), w(w), h(h),
		  crgs(0), scene(scene), cpu_lights(0), shadow_tracer(0), tracers(0), shadow_tracers(0) {
	}
	virtual ~cpu_pt() {
		set.basic_as<B, T>()->free_canonical_triangles(triangles);
	}

	virtual void activate(rta::rt_set *orig_set);
	virtual void compute();
	virtual void update();
	virtual bool progressive() { return true; }
	virtual bool in_progress();
};




#endif

