#include "config.h"
#include "hybrid-pt.h"

#include "raygen.h"
#include "material-wrapper.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <librta/intersect.h>
#include <libhyb/trav-util.h>

#if HAVE_LIBOSDINTERFACE == 1
#include <subdiv/osdi.h>
#endif

#include <omp.h>

using namespace std;
using namespace rta;
using namespace rta::cuda;
using namespace gi;
using namespace gi::cuda;


extern rta::cuda::material_t *gpu_materials;
extern rta::cuda::material_t *cpu_materials;
extern int material_count;
extern int idx_subd_material;
#if HAVE_LIBOSDINTERFACE == 1
extern std::vector<OSDI::Model*> subd_models;
#endif

#define DEBUG_PBRDF_FOR_SUBD 1 
#define SKYLIGHT_OFF 0

void hybrid_pt::activate(rt_set *orig_set) {
	if (activated) return;
	declare_variable<int>("pt/passes", 4);
	gi_algorithm::activate(orig_set);
	set = *orig_set;
	set.rt = set.rt->copy();
		
	nr_of_lights = gi::lights.size();
	cpu_lights = new gi::light[gi::lights.size()];	//gi::cuda::convert_and_upload_lights(nr_of_lights, overall_light_power);
	overall_light_power = 0;
	for (int i = 0; i < gi::lights.size(); ++i) {
		cpu_lights[i] = gi::lights[i];
		overall_light_power += cpu_lights[i].power;
	}
	
	triangles = set.basic_as<B, T>()->canonical_triangle_ptr();

// 	set.rgen = crgs = new rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials>(w, h);
	jitter = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	update_mt_pool(jitter);
	set.rgen = crgs = new rta::cuda::jittered_ray_generator(w, h, jitter);
	int bounces = 6;
	set.bouncer = pt = new hybrid_pt_bouncer<B, T>(w, h, cpu_materials, triangles, crgs, cpu_lights, nr_of_lights, bounces, vars["pt/passes"].int_val);
	
	gi::cuda::mt_pool3f pl = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	gi::cuda::mt_pool3f pp = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	update_mt_pool(pl);
	update_mt_pool(pp);
	pt->random_number_generator(pl, pp);


	// setup iterated tracers
	rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer> 
		*gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(set.rt);
	gpu_tracer->ray_bouncer(set.bouncer);
	gpu_tracer->ray_generator(set.rgen);
	tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::closest_hit_tracer>(gpu_tracer);

	shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer> 
		*shadow_gpu_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(shadow_tracer);
	shadow_tracers = new rta::cuda::iterated_gpu_tracers<B, T, rta::any_hit_tracer>(shadow_gpu_tracer);

	if (original_subd_set) {
		// subd closest hit
		rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>
			*subd_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::closest_hit_tracer>*>(original_subd_set->rt);
		subd_tracer->ray_generator(set.rgen);
		subd_tracer->ray_bouncer(set.bouncer);
		tracers->append_tracer(subd_tracer);
		// subd any hit
		rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>
			*subd_shadow_tracer = dynamic_cast<rta::cuda::gpu_raytracer<B, T, rta::any_hit_tracer>*>(subd_tracer->matching_any_hit_tracer());
		shadow_tracers->append_tracer(subd_shadow_tracer);
	}

// 	old tracer setup:
// 	tracer = new tandem_tracer<B, T>(dynamic_cast<basic_raytracer<B,T>*>(set.rt), 
// 									 dynamic_cast<basic_raytracer<B,T>*>(shadow_tracer));
	tracer = new tandem_tracer<B, T>(dynamic_cast<basic_raytracer<B,T>*>(tracers), 
									 dynamic_cast<basic_raytracer<B,T>*>(shadow_tracers));
	tracer->select_closest_hit_tracer();
	pt->register_tracers(tracer);
}

bool hybrid_pt::in_progress() {
	return (tracer && tracer->progressive_trace_running());
}

void hybrid_pt::update() {
	if (tracer->progressive_trace_running()) {
		tracer->trace_progressively(false);
		hybrid_pt_bouncer<B,T> *bouncer = dynamic_cast<hybrid_pt_bouncer<B, T>*>(set.bouncer);
		float3 *colors = bouncer->output_color;
		if (bouncer->path_len == 0) {
			update_mt_pool(jitter);	// a path is completed, so we generate new random numbers for the primary ray generator.
			gi::save_image("pt", bouncer->curr_path, w, h, colors);
		}
	}
}

void hybrid_pt::compute() {
		cout << "restarting progressive display" << endl;
		vec3f pos, dir, up;
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
		extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
		extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
		extract_up_vec3f_of_matrix(&up, lookat_matrix);
		if (nr_of_lights != gi::lights.size())
			throw std::runtime_error("number of lights changed in " "hybrid_pt::compute" 
									 " this exception is just for consistency to the gpu version.");
		overall_light_power = 0;
		for (int i = 0; i < gi::lights.size(); ++i) {
			cpu_lights[i] = gi::lights[i];
			overall_light_power += cpu_lights[i].power;
		}
		hybrid_pt_bouncer<B,T> *bouncer = dynamic_cast<hybrid_pt_bouncer<B, T>*>(set.bouncer);
		bouncer->overall_light_power = overall_light_power;
		bouncer->verbose = verbose;
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

		bouncer->path_samples = vars["pt/passes"].int_val;

		tracer->trace_progressively(true);
}
float3 evaluateSkyLight(gi::light *L, float3 &dir){
	if (!L) return make_float3(0,0,0);	// FIXME: black or white? doesnt matter, but I would leave black
	float3 *skylightData = L->skylight.data;
	float theta = acosf(dir.y);
	float phi = atan2f(dir.z, dir.x);
	if (phi < 0) phi += 2.0f*float(M_PI);
	float s = phi/(2.0f*float(M_PI));
	float t = theta/float(M_PI);
	int idx = int(t*L->skylight.h) * L->skylight.w + int(s*L->skylight.w);
	//TODO: This should actually not happen :(
	if(idx < 0 || idx >= L->skylight.h * L->skylight.w) {return make_float3(0.f,0.f,0.f);} //std::cerr<<"Error:Evaluate skylight: index "<<idx<<" and "<<dir.x<<","<<dir.y<<","<<dir.z<<"\n";//computed from "<<phi<<" and "<<theta<<" to "<<s<<","<<t<<"\n";
 	return skylightData[idx];

}	

void handle_invalid_intersection(int id, float3 *ray_orig,float3 *ray_dir, float* max_t,float3* throughput,float3 *col_accum,gi::light *skylight){
	float3 accSkylight = make_float3(0.f,0.f,0.f);
	float3 orgDir = ray_dir[id];
	if (max_t[id] == -1){}
	else{
		normalize_vec3f(&orgDir);
#if SKYLIGHT_OFF
		accSkylight = make_float3(1.f,1.f,1.f);
#else
		accSkylight = evaluateSkyLight(skylight,orgDir);
#endif
	}
	col_accum[id] += throughput[id] * accSkylight;
	ray_dir[id]  = make_float3(0,0,0);
	ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	max_t[id] = -1;
	throughput[id] = make_float3(0,0,0);


}
void compute_path_contribution_and_bounce(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t, float3 *ray_diff_org, float3 *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, float3 *uniform_random, float3 *throughput, float3 *col_accum,
										  float3 *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::light *skylight) {
	#pragma omp parallel for 
	for (int y = 0; y < h; ++y) {
		materialBRDF currentMaterial;
		for (int x = 0; x < w; ++x) {
			// general setup, early out if the path intersection is invalid.
			int2 gid = make_int2(x, y);
			int id = gid.y*w+gid.x;
			triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
			if (is.valid()) {
				float2 TC;
				float3 bc, P, N;
				material_t mat;
				rta::cuda::simple_triangle tri;
				float3 Tx, Ty;
				bool usePtexTexture = false;
				// check if we hit a triangle or a subd patch
				if ((is.ref & 0xff000000) == 0) {
					// load hit triangle and compute hitpoint geometry
					// we load the material, too, as it is stored in the triangle
					tri = triangles[is.ref];
					mat = mats[tri.material_index];
					is.barycentric_coord(&bc);
					barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
					barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
					barycentric_interpolation(&TC, &bc, &tri.ta, &tri.tb, &tri.tc);
					normalize_vec3f(&N);

					//TODO:CHECKME
					float3 dpos = tri.b - tri.a;
					float3 triTb3 = make_float3(tri.tb.x,tri.tb.y,1.0f);
					float3 triTa3 = make_float3(tri.ta.x,tri.ta.y,1.0f);
					float3 duv3 = triTb3 - triTa3;
					if(duv3.x == 0) {
						dpos = tri.c - tri.a;
						float3 triTb3 = make_float3(tri.tc.x,tri.tc.y,1.0f);
						duv3 = tri.c - tri.a;
						if(duv3.x == 0) duv3.x = 1.0f;
					}
					Tx = dpos * (1.0f/duv3.x);
					normalize_vec3f(&Tx);
					cross_vec3f(&Ty,&Tx,&N);
					normalize_vec3f(&Ty);
				}
				else {
#if HAVE_LIBOSDINTERFACE == 1
					// evaluate subd patch to get position and normal
					unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
					unsigned int ptexID = 0x00ffffff & is.ref;
					bool WITH_DISPLACEMENT = true;
					//!TODO:tangents are only written IF no displacement? Why?
					if (WITH_DISPLACEMENT)
						subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, true, (float*)&P, (float*)&N);
					else
						subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, false, (float*)&P, (float*)&N, (float*)&Tx, (float*)&Ty);
					// evaluate color and store it in the material as diffuse component
				
					if(idx_subd_material+modelidx < material_count) {
							mat = mats[idx_subd_material + modelidx];
					}else{
						std::cerr << "Warning: Could not open material at subd model idx " << modelidx <<". There are only " << material_count <<" materials present.\n";
						mat = mats[material_count-1];
					}
					usePtexTexture = true;
					 
#if DEBUG_PBRDF_FOR_SUBD
					//if DEBUG_BRDF we just take a constant color (faster than using ptex lookup)
					mat.diffuse_color = mat.parameters->color;
#else	
					//!slow but better with ram. 
					subd_models[modelidx]->EvalColor(ptexID, is.beta, is.gamma, (float*)&mat.diffuse_color);
#endif
					//!TODO Setup "correct" triangle for ray differentials, is this neccessary?
					tri.a = P; tri.b = P+Tx; tri.c = P+Ty;
					normalize_vec3f(&N);
					//tri.ta = du;
					//tri.tb = dv;
					//tri.tc = normalize_vec3f(du+dv);
#endif
				}
				
				// eval ray differentials (stored below)
				float3 upper_org = ray_diff_org[id],
					   upper_dir = ray_diff_dir[id],
					   right_org = ray_diff_org[w*h+id],
					   right_dir = ray_diff_dir[w*h+id];
				float3 upper_P, right_P;
				float2 upper_T, right_T;
				/* temporarily disabled ray differentials. (had to enable it to get correct texture lookup for .obj) Undefined behavior in case of Subd Models!
				 * we could just declare the triangle with the material above and load it in the triangle-branch.
				 * in the subd-brach we could get the tangents from the osdi lib and generate some triangle from it.
				 * the following call is actually just a plane intersection.*/
				triangle_intersection<rta::cuda::simple_triangle> other_is;
				// - upper ray
				intersect_tri_opt_nocheck(tri, (vec3f*)&upper_org, (vec3f*)&upper_dir, other_is);
				other_is.barycentric_coord(&bc);
				barycentric_interpolation(&upper_T, &bc, &tri.ta, &tri.tb, &tri.tc);
				barycentric_interpolation(&upper_P, &bc, &tri.a, &tri.b, &tri.c);
				// - right ray
				intersect_tri_opt_nocheck(tri, (vec3f*)&right_org, (vec3f*)&right_dir, other_is);
				other_is.barycentric_coord(&bc);
				barycentric_interpolation(&right_T, &bc, &tri.ta, &tri.tb, &tri.tc);
				barycentric_interpolation(&right_P, &bc, &tri.a, &tri.b, &tri.c);
				// - store origins
				ray_diff_org[id] = upper_P;
				ray_diff_org[w*h+id] = right_P;

				// load and evaluate material
				currentMaterial.init(mat.isPrincipledMaterial(),usePtexTexture,&mat, TC, upper_T, right_T, Tx, Ty);
				float3 org_dir = ray_dir[id];

				// add lighting to accumulation buffer
				normalize_vec3f(&org_dir);
				//bakcfacing check
				if((org_dir|N) > 0.0f) {
					handle_invalid_intersection(id, ray_orig, ray_dir, max_t, throughput,col_accum,skylight);
					continue;
				//	N = -1.0f*N;
				}
			
				//invert org dir to be consistent
				float3 inv_org_dir = -1.0f*org_dir;
				// attention: recycling of 'is'
				is = shadow_ti[id];
				float3 TP = throughput[id];
				if (!is.valid()) {
					float3 weight = potential_sample_contribution[id];
					// attention: we need the throughput *before* the bounce
					float3 curr = TP * weight;
					float3 light_dir = to_light[id];
					normalize_vec3f(&light_dir);
					// the whole geometric term is already computed in potential_sample_contribution.
					float3 brdf = currentMaterial.evaluate(inv_org_dir,light_dir,N);
					col_accum[id] +=  brdf *curr;
				}

				// compute next path segment by sampling the brdf
				float3 random = next_random3f(uniform_random, id);

				float3 T, B;				
				make_tangent_frame(N, T, B);
				normalize_vec3f(&T);
				normalize_vec3f(&B);

				float3 dir;
				bool reflection = false;
				//do only diffuse for now
				float pdf = 1.0f;
				if (reflection) {
					org_dir = transform_to_tangent_frame(org_dir, T, B, N);
					dir = org_dir;
					dir.z = -dir.z;
					dir = transform_from_tangent_frame(dir, T, B, N);
					// store reflection ray differentials. (todo: change to broadened cone)
					ray_diff_dir[id] = reflect(upper_dir, N);
					ray_diff_dir[w*h+id] = reflect(right_dir, N);
				}else{
					float3 inv_org_dir_ts = transform_to_tangent_frame(inv_org_dir,T,B,N);
					currentMaterial.sample(inv_org_dir_ts, dir, random, pdf);
					dir = transform_from_tangent_frame(dir,T,B,N);
					float3 brdf = currentMaterial.evaluate(inv_org_dir,dir,N);
					ray_diff_dir[id] = reflect(upper_dir, N);
					TP *= brdf;
				}
				float len = length_of_vector(dir);
				dir /= len;
				//direction of normal might be better 
				P += 0.01f * N; // I hope N is normalized :-)
				ray_orig[id] = P;
				ray_dir[id]  = dir;
				max_t[id]    = FLT_MAX;
				TP *= (1.0f/pdf);
				throughput[id] = TP;
				continue;
			}
			handle_invalid_intersection(id, ray_orig, ray_dir, max_t, throughput,col_accum,skylight);
			continue;
		}
	}
}


/* vim: set foldmethod=i*/

