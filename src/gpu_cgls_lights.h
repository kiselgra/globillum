#ifndef __GPU_CGLS_LIGHTS_H__ 
#define __GPU_CGLS_LIGHTS_H__ 

#include "direct-lighting.h"

namespace local {
	
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
		int nr_of_gpu_lights;
		rta::raytracer *shadow_tracer;
	public:
		gpu_cgls_lights(int w, int h, scene_ref scene, const std::string &name = "gpu_cgls_lights")
			: gi_algorithm(name), w(w), h(h),  /*TMP*/ hitpoints(w,h), normals(w,h),
			  collector(0), crgs(0), scene(scene), gpu_lights(0), shadow_tracer(0) {
		}

		void evaluate_material();
		void save(vec3f *out);

		virtual void activate(rta::rt_set *orig_set);
		virtual void compute();
		virtual void update();
		virtual bool progressive() { return true; }
	};

	class gpu_cgls_lights_dof : public gpu_cgls_lights {
		typedef rta::cuda::simple_aabb B;
		typedef rta::cuda::simple_triangle T;
	protected:
		float focus_distance, aperture, eye_to_lens;
		gi::cuda::mt_pool3f jitter;
	public:
		gpu_cgls_lights_dof(int w, int h, scene_ref scene, 
							float focus_distance, float aperture, float eye_to_lens, 
							const std::string &name = "gpu_cgls_lights_dof")
			: gpu_cgls_lights(w, h, scene, name), focus_distance(focus_distance), aperture(aperture), eye_to_lens(eye_to_lens) {
		}

		virtual void activate(rta::rt_set *orig_set);
		virtual void compute();
		virtual void update();
		virtual bool progressive() { return true; }
	};

}


#endif

