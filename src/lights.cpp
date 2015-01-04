#ifndef SCM_MAGIC_SNARFER
#include "lights.h"

#include <stdexcept>
#include <libcgl/scheme.h>
#include <librta/material.h>
#endif


using namespace std;

namespace gi {

	vector<light> lights;

	namespace cuda {
			
		light* convert_and_upload_lights(int &N, float &power) {
			N = 0;
			for (light &l : lights) {
				if (l.type == light::rect || l.type == light::sky)
					++N;
			}
			light *L;
			cout << "lights: " << N << endl;
			checked_cuda(cudaMalloc(&L, sizeof(light)*N));
			update_lights(L, N, power);
			return L;
		}

		void update_lights(light *data, int N, float &power) {
			vector<light> use_lights;
			power = 0;
			int n = 0;
			for (light &l : lights) {
				if (l.type == light::rect) {
					use_lights.push_back(l);
					power += l.power;
					++n;
				}
				else if (l.type == light::sky) {
					use_lights.push_back(l);
					light &nl = use_lights.back();
					checked_cuda(cudaMalloc(&nl.skylight.data, sizeof(float3)*nl.skylight.w*nl.skylight.h));
					checked_cuda(cudaMemcpy(nl.skylight.data, l.skylight.data, sizeof(float3)*nl.skylight.w*nl.skylight.h, cudaMemcpyHostToDevice));
					power += l.power;
					++n;
				}
			}
			if (n != N)
				throw std::runtime_error("number of lights changed in " "update_rectangular_area_lights");
			checked_cuda(cudaMemcpy(data, &use_lights[0], sizeof(light)*n, cudaMemcpyHostToDevice));
		}
		

	}

	// we just hijack rta's texture loading
	void load_skylight_data(sky_light *sl) {
		rta::texture tex(sl->map);
		sl->w = tex.w;
		sl->h = tex.h;
		sl->data = new float3[sl->w * sl->h];
		memcpy(sl->data, tex.data[0], sizeof(float3)*sl->w*sl->h);
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
	SCM_DEFINE(s_addrectlight, "add-rectlight", 7, 0, 0, (SCM name, SCM center, SCM dir, SCM up, SCM col, SCM w, SCM h), "internal function") {
		light l;
		l.type = light::rect;
		l.rectlight.center = scm_to_float3(center);
		l.rectlight.dir    = scm_to_float3(dir);
		l.rectlight.up     = scm_to_float3(up);
		l.rectlight.col    = scm_to_float3(col);
		l.rectlight.wh     = make_float2(scm_to_double(w), scm_to_double(h));
		l.power = l.rectlight.wh.x * l.rectlight.wh.y * (l.rectlight.col.x + l.rectlight.col.y + l.rectlight.col.z)*.333;
		lights.push_back(l);
		cout << "light '" << scm_to_locale_string(name) << "' has power " << l.power << endl;
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_addskylight, "add-skylight", 4, 0, 0, (SCM name, SCM mapname, SCM diameter, SCM scale), "internal function") {
		light l;
		l.type = light::sky;
		float s = scm_to_double(scale);
		float d = scm_to_double(diameter);
		char *map = scm_to_locale_string(mapname);
		l.skylight.scale = s;
		l.skylight.map = map;
		l.power = s*4*M_PI*(d*.5)*(d*.5);
		load_skylight_data(&l.skylight);
		lights.push_back(l);
		cout << "light '" << scm_to_locale_string(name) << "' has power " << l.power << endl;
		return SCM_BOOL_T;
	}

#ifndef SCM_MAGIC_SNARFER
	void register_scheme_functions_for_light_setup() {
		#include "lights.x"
	}
#endif

}

#endif

/* vim: set foldmethod=marker: */

