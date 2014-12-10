#ifndef SCM_MAGIC_SNARFER
#include "lights.h"
#endif

#include <stdexcept>
#include <libcgl/scheme.h>


using namespace std;

namespace gi {

	vector<light> lights;

	namespace cuda {
			
		light* convert_and_upload_lights(int &N) {
			N = 0;
			for (light &l : lights) {
				if (l.type == light::rect)
					++N;
			}
			light *L;
			cout << "lights: " << N << endl;
			checked_cuda(cudaMalloc(&L, sizeof(light)*N));
			update_lights(L, N);
			return L;
		}

		void update_lights(light *data, int N) {
			vector<light> use_lights;
			int n = 0;
			for (light &l : lights) {
				if (l.type == light::rect) {
					use_lights.push_back(l);
					++n;
				}
			}
			if (n != N)
				throw std::runtime_error("number of lights changed in " "update_rectangular_area_lights");
			checked_cuda(cudaMemcpy(data, &use_lights[0], sizeof(light)*n, cudaMemcpyHostToDevice));
		}

	}
}



#ifdef WITH_GUILE

#include <libguile.h>
#include <libcgl/scheme.h>

using namespace gi;

float3 scm_to_float3(SCM s) {
	vec3f v = scm_vec_to_vec3f(s);
	return make_float3(v.x, v.y, v.z);
}

extern "C" {
	SCM_DEFINE(s_addlight, "add-rectlight", 7, 0, 0, (SCM name, SCM center, SCM dir, SCM up, SCM col, SCM w, SCM h), "internal function") {
		light l;
		l.rectlight.center = scm_to_float3(center);
		l.rectlight.dir    = scm_to_float3(dir);
		l.rectlight.up     = scm_to_float3(up);
		l.rectlight.col    = scm_to_float3(col);
		lights.push_back(l);
	}

#ifndef SCM_MAGIC_SNARFER
	void register_scheme_functions_for_light_setup() {
		#include "lights.x"
	}
#endif

}

#endif

/* vim: set foldmethod=marker: */

