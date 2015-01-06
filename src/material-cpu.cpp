#include "material.h"

#include <librta/intersect.h>

#include <subdiv/osdi.h>
extern std::vector<OSDI::Model*> subd_models;

using namespace std;
using namespace rta;
using namespace rta::cuda;

namespace rta {
	void pixel_evaluate_material_bilin_lod_cpu(int2 gid, 
											   int w, int h, triangle_intersection<cuda::simple_triangle> *ti, 
											   cuda::simple_triangle *triangles, cuda::material_t *mats, float3 *dst, 
											   float3 *ray_org, float3 *ray_dir, float3 *ray_diff_org, float3 *ray_diff_dir, 
											   float3 background) {
		triangle_intersection<cuda::simple_triangle> is = ti[gid.y*w+gid.x];
		float3 out = background;
		if (is.valid()) {
			if (is.ref & 0xFF000000) {
				float3 N, P;
				unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
				unsigned int ptexID = 0x00ffffff & is.ref;
				subd_models[modelidx]->EvalColor(ptexID, is.beta, is.gamma, (float*)&out);
// 				subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, true, (float*)&P, (float*)&N);
// 				out = N;
			}
			else {
				cuda::simple_triangle tri = triangles[is.ref];
				material_t mat = mats[tri.material_index];
				out = mat.diffuse_color;
				if (mat.diffuse_texture) {
					float3 bc; 
					is.barycentric_coord(&bc);
					// tex coord
					const float2 &ta = tri.ta;
					const float2 &tb = tri.tb;
					const float2 &tc = tri.tc;
					float2 T;
					barycentric_interpolation(&T, &bc, &ta, &tb, &tc);
					// normal
					const float3 &na = tri.na;
					const float3 &nb = tri.nb;
					const float3 &nc = tri.nc;
					float3 N;
					barycentric_interpolation(&N, &bc, &na, &nb, &nc);
					// eval other rays
					// - upper ray
					float3 other_org = ray_diff_org[gid.y*w+gid.x];
					float3 other_dir = ray_diff_dir[gid.y*w+gid.x];
					triangle_intersection<cuda::simple_triangle> other_is;
					intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
					float2 other_T;
					float3 other_bc;
					other_is.barycentric_coord(&other_bc);
					barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
					float diff_x = fabsf(T.x - other_T.x);
					float diff_y = fabsf(T.y - other_T.y);
					// - right ray
					other_org = ray_diff_org[w*h+gid.y*w+gid.x];
					other_dir = ray_diff_dir[w*h+gid.y*w+gid.x];
					intersect_tri_opt_nocheck(tri, (vec3f*)&other_org, (vec3f*)&other_dir, other_is);
					other_is.barycentric_coord(&other_bc);
					barycentric_interpolation(&other_T, &other_bc, &ta, &tb, &tc);
					diff_x = fmaxf(fabsf(T.x - other_T.x), diff_x);
					diff_y = fmaxf(fabsf(T.y - other_T.y), diff_y);
					float diff = fmaxf(diff_x, diff_y);
					float3 tex = mat.diffuse_texture->sample_bilin_lod(T.x, T.y, diff);
					out.x *= tex.x;
					out.y *= tex.y;
					out.z *= tex.z;
				}
			}
		}
		dst[gid.y*w+gid.x] = out;
	}
	void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
						   cuda::material_t *mats, float3 *dst, float3 *ray_org, float3 *ray_dir, 
						   float3 *ray_diff_org, float3 *ray_diff_dir, float3 background) {
		#pragma omp parallel for
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				pixel_evaluate_material_bilin_lod_cpu(make_int2(x, y), 
													  w, h, ti, triangles, mats, dst, 
													  (float3*)ray_org, (float3*)ray_dir, 
													  (float3*)ray_diff_org, (float3*)ray_diff_dir, background);
			}
		}
	}
}

/* vim: set foldmethod=marker: */

