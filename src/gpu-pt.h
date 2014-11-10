#ifndef __GI_GPU_PT_H__ 
#define __GI_GPU_PT_H__ 

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"	// contains gpu_material_evaluator.
#include "arealight-sampler.h"
#include "gpu-pt-kernels.h"

#include "rayvis.h"

#include <iostream>

template<typename _box_t, typename _tri_t> struct gpu_pt_bouncer : public local::gpu_material_evaluator<forward_traits> {
	declare_traits_types;
	rta::cuda::cgls::rect_light *lights;
	int nr_of_lights;
	gi::cuda::halton_pool2f uniform_random_numbers;
	uint w, h;
	int curr_bounce;
	int path_len;
	int max_path_len;
	rta::triangle_intersection<rta::cuda::simple_triangle> *path_intersections,
	                                                       *shadow_intersections;
	float3 *output_color, *throughput, *potential_sample_contribution;

	gpu_pt_bouncer(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles,
				   rta::cuda::cam_ray_generator_shirley *crgs, rta::cuda::cgls::rect_light *lights, int nr_of_lights,
				   gi::cuda::halton_pool2f rnd, int max_path_len)
	: local::gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs),
	  lights(lights), nr_of_lights(nr_of_lights), uniform_random_numbers(rnd), w(w), h(h),
	  curr_bounce(0), path_len(0), max_path_len(max_path_len), output_color(0) {
		checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&throughput, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&potential_sample_contribution, sizeof(float3)*w*h));
		checked_cuda(cudaMalloc(&shadow_intersections, sizeof(rta::triangle_intersection<rta::cuda::simple_triangle>)*w*h));
		path_intersections = this->gpu_last_intersection;
	}
	~gpu_pt_bouncer() {
		checked_cuda(cudaFree(output_color));
		checked_cuda(cudaFree(shadow_intersections));
	}
	virtual void new_pass() {
		curr_bounce = path_len = 0;
		this->gpu_last_intersection = path_intersections;
		reset_gpu_buffer(throughput, w, h, make_float3(1,1,1));
		restart_rayvis();
	}
	virtual void setup_new_arealight_sample() {
		rta::cuda::cgls::generate_rectlight_sample(this->w, this->h, lights, nr_of_lights, 
												   this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
												   path_intersections, this->tri_ptr, uniform_random_numbers, potential_sample_contribution, 
												   curr_bounce);	// curr_bounce is random-offset
	}
	virtual void setup_new_path_sample() {
		generate_random_path_sample(this->w, this->h, this->crgs->gpu_origin, this->crgs->gpu_direction, this->crgs->gpu_maxt,
									path_intersections/* last intersection*/, this->tri_ptr, uniform_random_numbers, curr_bounce, throughput);
	}
	virtual void integrate_light_sample() {
		rta::cuda::cgls::integrate_light_sample(this->w, this->h, this->gpu_last_intersection, potential_sample_contribution,
												this->material_colors, throughput, output_color, curr_bounce-1);
	}
	virtual void bounce() {
		bool compute_light_sample = false;
		bool compute_path_segment = false;
			
		std::cout << "bounce " << curr_bounce << std::endl;

		if (curr_bounce == 0) {
			std::cout << " - eval mat" << std::endl;
			this->evaluate_material();
// 			compute_light_sample = true;

			vec3f campos = this->crgs->position;
			std::cout << " - adding camera position as vertex" << std::endl;
			add_vertex_to_all_rays(make_float3(campos.x, campos.y, campos.z));
			std::cout << " - adding primary hit as vertex" << std::endl;
			add_intersections_to_rays(this->w, this->h, this->gpu_last_intersection, this->tri_ptr);
// 			compute_light_sample = true;
			compute_path_segment = true;
		}
		else {
			if (this->gpu_last_intersection == shadow_intersections) {
				std::cout << " - integrating light sample" << std::endl;
				integrate_light_sample();
				compute_path_segment = true;
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
		}
		if (compute_path_segment) {
			std::cout << " - computing new path sample" << std::endl;
			setup_new_path_sample();
			this->gpu_last_intersection = path_intersections;
		}
		++curr_bounce;
	}
	virtual bool trace_further_bounces() {
// 		std::cout<<"cb: " << curr_bounce << std::endl;
		return curr_bounce < 2;
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
	rta::cuda::cam_ray_generator_shirley *crgs;
	rta::rt_set set;
	scene_ref scene;
	rta::cuda::cgls::rect_light *gpu_rect_lights;
	int nr_of_gpu_rect_lights;
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

