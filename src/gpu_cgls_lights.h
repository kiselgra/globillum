#ifndef __GPU_CGLS_LIGHTS_H__ 
#define __GPU_CGLS_LIGHTS_H__ 

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
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		gpu_material_evaluator(uint w, uint h, rta::cuda::material_t *materials, rta::cuda::simple_triangle *triangles, 
							   rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs)
			: rta::cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), material_colors(0), tri_ptr(triangles),
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
										 crgs->gpu_origin, crgs->gpu_direction, crgs->differentials_origin, crgs->differentials_direction);
		}
		virtual void bounce() {
			evaluate_material();
		}
		virtual std::string identification() {
			return "evaluate first-hit material on gpu.";
		}
	};


	class gpu_cgls_lights : public gi_algorithm {
		typedef rta::cuda::simple_aabb B;
		typedef rta::cuda::simple_triangle T;
	protected:
		int w, h;
		rta::cuda::primary_intersection_collector<B, T> *collector;
		rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials> *crgs;
		rta::rt_set set;
		rta::image<vec3f, 1> hitpoints, normals;
		scene_ref scene;
		rta::cuda::cgls::light *gpu_lights;
		rta::cuda::cgls::rect_light *gpu_rect_lights;
		int nr_of_gpu_lights, nr_of_gpu_rect_lights;
		rta::raytracer *shadow_tracer;
	public:
		gpu_cgls_lights(int w, int h, scene_ref scene, const std::string &name = "gpu_cgls_lights")
			: gi_algorithm(name), w(w), h(h),  /*TMP*/ hitpoints(w,h), normals(w,h),
			  collector(0), crgs(0), scene(scene), gpu_lights(0), gpu_rect_lights(0), shadow_tracer(0) {
		}

		void evaluate_material();
		void save(vec3f *out);

		virtual void activate(rta::rt_set *orig_set);
		virtual void compute();
		virtual void update();
		virtual bool progressive() { return true; }
	};

}


#endif

