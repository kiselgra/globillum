#include "hybrid-pt.h"

#include "raygen.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <librta/intersect.h>
#include <libhyb/trav-util.h>

#include <omp.h>

using namespace std;
using namespace rta;
using namespace rta::cuda;
using namespace gi;
using namespace gi::cuda;


extern rta::cuda::material_t *gpu_materials;
extern rta::cuda::material_t *cpu_materials;

//define only one of these
#define ALL_MATERIAL_LAMBERT 1 // if defined this will be grey lambert.
#define ALL_MATERIAL_BLINNPHONG 0

void hybrid_pt::activate(rt_set *orig_set) {
	if (activated) return;
	declare_variable<int>("pt/passes", 32);
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
	int bounces = 2;
	set.bouncer = pt = new hybrid_pt_bouncer<B, T>(w, h, cpu_materials, triangles, crgs, cpu_lights, nr_of_lights, bounces, vars["pt/passes"].int_val);
	
	gi::cuda::mt_pool3f pl = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	gi::cuda::mt_pool3f pp = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	update_mt_pool(pl);
	update_mt_pool(pp);
	pt->random_number_generator(pl, pp);
	
	set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
	set.basic_rt<B, T>()->ray_generator(set.rgen);
	shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	tracer = new tandem_tracer<B, T>(dynamic_cast<basic_raytracer<B,T>*>(set.rt), 
									 dynamic_cast<basic_raytracer<B,T>*>(shadow_tracer));
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
	
void compute_path_contribution_and_bounce(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t, float3 *ray_diff_org, float3 *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, float3 *uniform_random, float3 *throughput, float3 *col_accum,
										  float3 *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution) {
	#pragma omp parallel for 
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			// general setup, early out if the path intersection is invalid.
			int2 gid = make_int2(x, y);
			int id = gid.y*w+gid.x;
			triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
			if (is.valid()) {
				// load hit triangle and compute hitpoint geometry
				rta::cuda::simple_triangle tri = triangles[is.ref];
				float2 TC;
				float3 bc, P, N;
				is.barycentric_coord(&bc);
				barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);
				barycentric_interpolation(&N, &bc, &tri.na, &tri.nb, &tri.nc);
				barycentric_interpolation(&TC, &bc, &tri.ta, &tri.tb, &tri.tc);
				normalize_vec3f(&N);

				// eval ray differentials (stored below)
				float3 upper_org = ray_diff_org[id],
					   upper_dir = ray_diff_dir[id],
					   right_org = ray_diff_org[w*h+id],
					   right_dir = ray_diff_dir[w*h+id];
				triangle_intersection<rta::cuda::simple_triangle> other_is;
				float3 upper_P, right_P;
				float2 upper_T, right_T;
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
				material_t mat = mats[tri.material_index];
				#if ALL_MATERIAL_LAMBERT
				Material *lambertM = (Material*)new LambertianMaterial(&mat);
				#elif ALL_MATERIAL_BLINNPHONG
			 	Material *blinnM = (Material*) new BlinnMaterial(&mat, TC, upper_T, right_T);
				#endif
				// This has been moved to material.h
				/*float3 diffuse = mat.diffuse_color;
				float3 specular = mat.specular_color;
				if (mat.diffuse_texture || mat.specular_texture) {
					float diff_x = fabsf(TC.x - upper_T.x);
					float diff_y = fabsf(TC.y - upper_T.y);
					diff_x = fmaxf(fabsf(TC.x - right_T.x), diff_x);
					diff_y = fmaxf(fabsf(TC.y - right_T.y), diff_y);
					float diff = fmaxf(diff_x, diff_y);
					if (mat.diffuse_texture)
						diffuse *= mat.diffuse_texture->sample_bilin_lod(TC.x, TC.y, diff);
					if (mat.specular_texture)
						specular *= mat.specular_texture->sample_bilin_lod(TC.x, TC.y, diff);
				}
				float sum = diffuse.x+diffuse.y+diffuse.z+specular.x+specular.y+specular.z;;
				if (sum > 1.0f) {
					diffuse /= sum;
					specular /= sum;
				}*/

				// add lighting to accumulation buffer
				float3 org_dir = ray_dir[id];
				normalize_vec3f(&org_dir);
				//float3 R = reflect(org_dir, N);
				// attention: recycling of 'is'
				is = shadow_ti[id];
				float3 TP = throughput[id];
				if (!is.valid()) {
					float3 prev = col_accum[id];
					float3 weight = potential_sample_contribution[id];
					// attention: we need the throughput *before* the bounce
					float3 curr = TP * weight;
					float3 light_dir = to_light[id];
					normalize_vec3f(&light_dir);
					// the whole geometric term is already computed in potential_sample_contribution.
					//moved to material evaluate function float3 brdf = diffuse * float(M_1_PI)	+ (shininess + 1)*specular * 0.5 * M_1_PI * pow(fmaxf((R|light_dir), 0.0f), shininess);
					float3 brdf = make_float3(0.f,0.f,0.f);
					#if ALL_MATERIAL_LAMBERT
					brdf = lambertM->evaluate(org_dir,light_dir,N);
					#elif ALL_MATERIAL_BLINNPHONG
					brdf = blinnM->evaluate(org_dir,light_dir,N);
					#endif
					col_accum[id] = prev + brdf * curr;
				}

				// compute next path segment by sampling the brdf
				float3 random = next_random3f(uniform_random, id);
				/*float pd = diffuse.x+diffuse.y+diffuse.z;
				float ps = specular.x+specular.y+specular.z;
				if (pd + ps > 1) {
					pd /= pd+ps;
					ps /= pd+ps;
				}
				bool diffuse_bounce = false,specular_bounce = false;
				float P_component;
				if (random.z <= pd) {
					diffuse_bounce = true;
					P_component = pd;
				}
				else if (random.z <= pd+ps) {
					specular_bounce = true;
					P_component = ps;
				}*/

				float3 T, B;				
				make_tangent_frame(N, T, B);
				normalize_vec3f(&T);
				normalize_vec3f(&B);

				float3 dir;
				float3 use_color = make_float3(1,1,1);
				//float n;
				//float omega_z;
				bool reflection = false;
				//do only diffuse for now
				bool specular_bounce = false;
				bool diffuse_bounce = true;
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
					#if ALL_MATERIAL_LAMBERT
					lambertM->sample(org_dir,dir,random,N,pdf);
					#elif ALL_MATERIAL_BLINNPHONG
					blinnM->sample(org_dir,dir,random,N,pdf);
					#endif

					dir = transform_from_tangent_frame(dir,T,B,N);
					float3 brdf = make_float3(0.f,0.f,0.f);

					#if ALL_MATERIAL_LAMBERT
					brdf = lambertM->evaluate(org_dir,dir,N);
					#elif ALL_MATERIAL_BLINNPHONG
					brdf = blinnM->evaluate(org_dir,dir,N);
					#endif

					ray_diff_dir[w*h+id] = reflect(right_dir, N);
					ray_diff_dir[id] = reflect(upper_dir, N);
					TP *= brdf;
					use_color = brdf;
				}/*
				else if (diffuse_bounce) {
					//n = 1.0f;
					//dir.z = powf(random.x, 1.0f/(n+1.0f));
					//dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*random.y);
					//dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*random.y);
					//omega_z = dir.z;
					lambertM->sample(org_dir, dir, random, N, pdf);
					dir = transform_from_tangent_frame(dir, T, B, N);
					float3 brdf = lambertM->evaluate(org_dir,dir,N);
					// store reflection ray differentials. (todo: change to broadened cone)
					ray_diff_dir[id] = reflect(upper_dir, N);
					ray_diff_dir[w*h+id] = reflect(right_dir, N);
					use_color = diffuse;
					//float3 brdf = fabsf(dir|N) * diffuse * float(M_1_PI);// * fmaxf((N|-org_dir), 0.0f);
					TP *= brdf;
				}
				else if (specular_bounce) {
					make_tangent_frame(R, T, B);
					n = shininess;
					dir.z = powf(random.x, 1.0f/(n+1.0f));
					dir.x = sqrtf(1.0f-dir.z*dir.z) * cosf(2.0f*float(M_PI)*random.y);
					dir.y = sqrtf(1.0f-dir.z*dir.z) * sinf(2.0f*float(M_PI)*random.y);
					omega_z = dir.z;
					dir = transform_from_tangent_frame(dir, T, B, R);
					// store reflection ray differentials.
					ray_diff_dir[id] = reflect(upper_dir, N);
					ray_diff_dir[w*h+id] = reflect(right_dir, N);
					if ((dir|N)<0) specular_bounce = false;
					use_color = specular;
					float3 brdf = fabsf(dir|N) * (shininess + 1)*specular * 0.5 * M_1_PI * pow(fmaxf((R|dir), 0.0f), shininess);
					TP *= brdf;
				}*/
				#if ALL_MATERIAL_LAMBERT
				delete lambertM;
				#elif ALL_MATERIAL_BLINNPHONG
				delete blinnM;
				#endif
				if (diffuse_bounce||specular_bounce) {
					float len = length_of_vector(dir);
					dir /= len;

					//P += 0.01*dir;
					//direction of normal might be better 
					P += 0.01f * N; // I hope N is normalized :-)
					ray_orig[id] = P;
					ray_dir[id]  = dir;
					max_t[id]    = FLT_MAX;
					//TP *= (1.0f/P_component) * ((2.0f*float(M_PI))/((n+1.0f)*pow(omega_z,n)));
					TP *= (1.0f/pdf);
					throughput[id] = TP;
					// 				throughput[id] = make_float3(0,0,0);
					continue;
				}
			}
			// fall through for absorption and invalid intersection
			ray_dir[id]  = make_float3(0,0,0);
			ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			max_t[id] = -1;
			throughput[id] = make_float3(0,0,0);
		}
	}
}


/* vim: set foldmethod=i*/

