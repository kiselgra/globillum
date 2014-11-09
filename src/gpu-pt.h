#ifndef __GI_GPU_PT_H__ 
#define __GI_GPU_PT_H__ 

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"	// contains gpu_material_evaluator.

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
	float3 *output_color;

	gpu_pt_bouncer(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles,
				   rta::cuda::cam_ray_generator_shirley *crgs, rta::cuda::cgls::rect_light *lights, int nr_of_lights,
				   gi::cuda::halton_pool2f rnd, int max_path_len)
	: local::gpu_material_evaluator<forward_traits>(w, h, materials, triangles, crgs),
	  lights(lights), nr_of_lights(nr_of_lights), uniform_random_numbers(rnd), w(w), h(h),
	  curr_bounce(0), path_len(0), max_path_len(max_path_len), output_color(0) {
		checked_cuda(cudaMalloc(&output_color, sizeof(float3)*w*h));
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
	}
	virtual void bounce() {
		this->evaluate_material();
	}
	virtual bool trace_further_bounces() {
		return false;
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

