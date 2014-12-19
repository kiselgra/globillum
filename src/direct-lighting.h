#ifndef __DIRECT_LIGHTING_H__ 
#define __DIRECT_LIGHTING_H__ 

#include "gi_algorithm.h"

#include "arealight-sampler.h"
#include "cgls-lights.h"
#include "material.h"

#include <libcgls/scene.h>
#include <librta/cuda.h>

namespace local {
	
	template<typename _box_t, typename _tri_t> struct gpu_material_evaluator : public rta::cuda::gpu_ray_bouncer<forward_traits> {
		declare_traits_types;
		rta::cuda::material_t *materials;
		rta::cuda::simple_triangle *tri_ptr;
		float3 *material_colors;
		float3 background;
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		gpu_material_evaluator(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles, 
							   rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs)
			: rta::cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), material_colors(0), tri_ptr(triangles),
			  crgs(crgs), background(make_float3(0,0,0)) {
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
										 crgs->gpu_origin, crgs->gpu_direction, crgs->differentials_origin, crgs->differentials_direction, background);
		}
		virtual void bounce() {
			evaluate_material();
		}
		virtual std::string identification() {
			return "evaluate first-hit material on gpu.";
		}
	};

	template<typename _box_t, typename _tri_t> struct cpu_material_evaluator : public rta::cpu_ray_bouncer<forward_traits> {
		declare_traits_types;
		rta::cuda::material_t *materials;
		rta::cuda::simple_triangle *tri_ptr;
		float3 *material_colors;
		float3 background;
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		cpu_material_evaluator(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles, 
							   rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs)
			: rta::cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), material_colors(0), tri_ptr(triangles),
			  crgs(crgs), background(make_float3(0,0,0)) {
// 			checked_cuda(cudaMalloc(&material_colors, sizeof(float3)*w*h));
		}
		~cpu_material_evaluator() {
// 			checked_cuda(cudaFree(material_colors));
		}
		virtual bool trace_further_bounces() {
			return false;
		}
		virtual void evaluate_material() {
// 			rta::cuda::evaluate_material(this->w, this->h, this->gpu_last_intersection, tri_ptr, materials, material_colors, 
// 										 crgs->gpu_origin, crgs->gpu_direction, crgs->differentials_origin, crgs->differentials_direction, background);
		}
		virtual void bounce() {
			evaluate_material();
		}
		virtual std::string identification() {
			return "evaluate first-hit material on cpu.";
		}
	};


	class gpu_arealight_sampler : public gi_algorithm {
		typedef rta::cuda::simple_aabb B;
		typedef rta::cuda::simple_triangle T;
	protected:
		int w, h;
		rta::cuda::primary_intersection_collector<B, T> *collector;
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		rta::rt_set set;
		rta::image<vec3f, 1> hitpoints, normals;
		scene_ref scene;
		gi::light *gpu_lights;
		int nr_of_gpu_lights;
		rta::raytracer *shadow_tracer;
		float overall_light_power;
	public:
		gpu_arealight_sampler(int w, int h, scene_ref scene, const std::string &name = "gpu_area_lights")
			: gi_algorithm(name), w(w), h(h),  /*TMP*/ hitpoints(w,h), normals(w,h),
			  collector(0), crgs(0), scene(scene), gpu_lights(0), shadow_tracer(0), overall_light_power(0) {
		}

		void evaluate_material();
		void save(vec3f *out);

		virtual void activate(rta::rt_set *orig_set);
		virtual bool in_progress();
		virtual void compute();
		virtual void update();
		virtual bool progressive() { return true; }
		virtual void light_samples(int n);
	};
	
	template<typename _box_t, typename _tri_t> struct hybrid_arealight_evaluator : public gpu_material_evaluator<forward_traits> {
		declare_traits_types;
		typedef rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> crgs_with_diffs;
		gi::light *lights;
		int nr_of_lights;
		gi::cuda::mt_pool3f uniform_random_numbers;
		float3 *potential_sample_contribution; 
		int curr_bounce;
		int samples;
		float3 *output_color;
		float overall_light_power;
		rta::triangle_intersection<rta::cuda::simple_triangle> *primary_intersection;
		hybrid_arealight_evaluator(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles, 
								   crgs_with_diffs *crgs, gi::light *lights, int nr_of_lights,
								   gi::cuda::mt_pool3f rnd, int samples);
		~hybrid_arealight_evaluator();
		virtual void new_pass();
		virtual bool trace_further_bounces();
		virtual void setup_new_arealight_sample();
		virtual void integrate_light_sample();
		virtual void bounce();
		virtual std::string identification() {
			return "cpu area light shader to be used with a gpu tracer (might work with cpu tracers, too).";
		}
	};


	class hybrid_arealight_sampler : public gi_algorithm {
		typedef rta::cuda::simple_aabb B;
		typedef rta::cuda::simple_triangle T;
	protected:
		int w, h;
		rta::cuda::primary_intersection_downloader<B, T, hybrid_arealight_evaluator<B, T>> *collector;
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		rta::rt_set set;
		rta::image<vec3f, 1> hitpoints, normals;
		scene_ref scene;
		gi::light *cpu_lights;
		int nr_of_lights;
		rta::raytracer *shadow_tracer;
		float overall_light_power;
	public:
		hybrid_arealight_sampler(int w, int h, scene_ref scene, const std::string &name = "hybrid_area_lights")
			: gi_algorithm(name), w(w), h(h),  /*TMP*/ hitpoints(w,h), normals(w,h),
			  collector(0), crgs(0), scene(scene), cpu_lights(0), shadow_tracer(0), overall_light_power(0) {
		}

		void evaluate_material();
		void save(vec3f *out);

		virtual void activate(rta::rt_set *orig_set);
		virtual bool in_progress();
		virtual void compute();
		virtual void update();
		virtual bool progressive() { return true; }
		virtual void light_samples(int n);
	};

}


#endif

