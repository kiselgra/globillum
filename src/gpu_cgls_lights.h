#ifndef __GPU_CGLS_LIGHTS_H__ 
#define __GPU_CGLS_LIGHTS_H__ 

#include "gi_algorithm.h"

#include <librta/cuda.h>

namespace local {

	class gpu_cgls_lights : public gi_algorithm {
		typedef rta::cuda::simple_aabb B;
		typedef rta::cuda::simple_triangle T;
	protected:
		int w, h;
		rta::cuda::primary_intersection_collector<B, T> *collector;
		rta::cuda::cam_ray_generator_shirley *crgs;
		rta::rt_set set;
		rta::image<vec3f, 1> hitpoints, normals;
	public:
		gpu_cgls_lights(int w, int h, const std::string &name = "gpu_cgls_lights")
			: gi_algorithm(name), w(w), h(h),  /*TMP*/ hitpoints(w,h), normals(w,h),
			  collector(0), crgs(0) {
		}

		void evaluate_material();
		void save(vec3f *out);

		virtual void activate(rta::rt_set *orig_set);
		virtual void compute();
	};

}


#endif

