#include "arealight-sampler.h"

#include "util.h"

#include <libhyb/trav-util.h>
#include <subdiv/osdi.h>

#include <vector>
#include <iostream>

using namespace std;
using namespace rta;
using namespace gi;
#define USE_SKY_SAMPLING
extern vector<OSDI::Model*> subd_models;

namespace rta {
	template<typename rng_t>
	void pixel_generate_arealight_sample_cpu(int2 gid, 
											 int w, int h, gi::light *lights, int nr_of_lights, float overall_power,
											 float3 *ray_orig, float3 *ray_dir, float *max_t,
											 triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
											 rng_t uniform01, float3 *potential_sample_contribution) {
		int id = gid.y*w+gid.x;
		float3 rnd = gi::next_random3f(uniform01, id);
		float choice = rnd.z*overall_power;
		float light_acc = 0;
		int light = 0;
		while (choice > light_acc+lights[light].power && light < nr_of_lights) {
			light_acc += lights[light].power;
			++light;
		}

		triangle_intersection<cuda::simple_triangle> is = ti[id];
		if (is.valid()) {
			float3 bc; 
			float3 P, N;
			if ((is.ref & 0xff000000) == 0) {
				cuda::simple_triangle tri = triangles[is.ref];
				is.barycentric_coord(&bc);
				barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
				barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
			}
			else {
				unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
				unsigned int ptexID = 0x00ffffff & is.ref;
				bool WITH_DISPLACEMENT = true;
				if (WITH_DISPLACEMENT)
					subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, true, (float*)&P, (float*)&N);
				else
					subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, false, (float*)&P, (float*)&N);
			}			

			float3 contribution = make_float3(0.0f,0.0f,0.0f);
			float3 dir;
			float len;

			if (lights[light].type == gi::light::rect) {
				float3 light_pos = lights[light].rectlight.center;
				float3 light_dir = lights[light].rectlight.dir;
				float3 right = make_tangential(make_float3(1,0,0), light_dir);
				float3 up = make_tangential(make_float3(0,1,0), light_dir);
				// 						float3 right = make_float3(1,0,0);
				// 						float3 up = make_float3(0,0,1);
				float2 offset = make_float2((rnd.x - 0.5f) * lights[light].rectlight.wh.x,
											(rnd.y - 0.5f) * lights[light].rectlight.wh.y);
				float3 light_sample = light_pos + offset.x * right + offset.y * up;
				dir = light_sample - P;
				len = length_of_vector(dir);
				dir /= len;
				float ndotl = fmaxf((N|dir), 0.0f);
				float light_cos = fmaxf((light_dir|-dir), 0.0f);
				float factor = lights[light].rectlight.wh.x * lights[light].rectlight.wh.y * ndotl * light_cos / (len*len);
				contribution = lights[light].rectlight.col * factor;
			}
			else if (lights[light].type == gi::light::sky) {
				len = FLT_MAX;
				sky_light &sl = lights[light].skylight;
			#ifdef USE_SKY_SAMPLING
		                 float outPdf = 1.0f;
                                 float3 L = sl.sample(rnd.x, rnd.y, outPdf, dir);
                                 dir = make_tangential(dir,N);
                                 float a = 1.0f/(outPdf);
				contribution = sl.scale * L * a * fabs(dir|N);
                        #else
				float sq = sqrtf(1-rnd.x*rnd.x);
				dir.x = sq * cosf(2.0f*float(M_PI)*rnd.y);
				dir.y = sq * sinf(2.0f*float(M_PI)*rnd.y);
				dir.z = rnd.x;
				dir = make_tangential(dir, N);
				len = FLT_MAX;
				float theta = acosf(dir.y);
				float phi = atan2f(dir.z, dir.x);
				// 						float theta = acosf(dir.z);
				// 						float phi = atan2f(dir.y, dir.x);
				if (phi < 0) phi += 2.0f*float(M_PI);
				float s = phi/(2.0f*float(M_PI));
				float t = theta/float(M_PI);
				contribution = sl.scale * sl.data[int(t*sl.h) * sl.w + int(s*sl.w)] * (dir|N);
			#endif
			}

			normalize_vec3f(&dir);
			P += 0.9*dir;
			ray_orig[id] = P;
			ray_dir[id]  = dir;
			max_t[id]    = len;
			potential_sample_contribution[id] = contribution * (overall_power/lights[light].power);
		}
		else {
			float3 dir = ray_dir[id];
			ray_dir[id]  = make_float3(0,0,0);
			ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			max_t[id] = -1;
			// potential_sample_contribution[id] = make_float3(0, 0, 0);
			float3 Le = make_float3(0.0f,0.0f,0.0f);
			int i = 0;
			for (i = nr_of_lights-1; i >= 0; i--)
				if (lights[i].type == gi::light::sky)
					break;
			if (i >= 0) {
				float theta = acosf(dir.y);
				float phi = atan2f(dir.z, dir.x);
				if (phi < 0) phi += 2.0f*float(M_PI);
				float s = phi/(2.0f*float(M_PI));
				float t = theta/float(M_PI);
				sky_light &sl = lights[i].skylight;
				Le = sl.scale * sl.data[int(t*sl.h) * sl.w + int(s*sl.w)];
			}
			potential_sample_contribution[id] = Le;
		}
	}
	void generate_arealight_sample(int w, int h, gi::light *lights, int nr_of_lights, float overall_power,
								   float3 *ray_orig, float3 *ray_dir, float *max_t,
								   triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles,
								   float3 *uniform01, float3 *potential_sample_contribution) {
#pragma omp prallel for schedule(dynamic, 1)
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				pixel_generate_arealight_sample_cpu(make_int2(x, y),
													w, h, lights, nr_of_lights, overall_power, ray_orig, ray_dir, max_t, 
													ti, triangles, uniform01, potential_sample_contribution);
			}
		}
	}


}

/* vim: set foldmethod=marker: */

