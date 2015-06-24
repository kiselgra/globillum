#include "config.h"
#include "hybrid-pt.h"

#include "raygen.h"
#include "material-wrapper.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <librta/intersect.h>
#include <libhyb/trav-util.h>

#if HAVE_LIBOSDINTERFACE == 1
#include <rta-0.0.1/subdiv/osdi.h>
#endif

#include <omp.h>
#include "colormap.h"
#include <libcgl/wall-time.h>

using namespace std;
using namespace rta;
using namespace rta::cuda;
using namespace gi;
using namespace gi::cuda;


extern rta::cuda::material_t *gpu_materials;
extern rta::cuda::material_t *cpu_materials;
extern int material_count;
extern int idx_subd_material;
extern int curr_frame;
extern double time_adaptive_subd_eval;
#if HAVE_LIBOSDINTERFACE == 1
extern std::vector<OSDI::Model*> subd_models;
#endif

#define RENDER_UVS 0  		//	render uvs image.
#define DIFF_ERROR_IMAGE 0	// 	render diff error image.
#define RENDER_PATTERN 0

#if RENDER_UVS || DIFF_ERROR_IMAGE || RENDER_PATTERN
	#define NO_BACKGROUND
#endif

#define NO_BACKGROUND // i.e. for rendering teaser image.

//evaluate the time necessary for CPU-FAS.
#define TIME_ADAPTIVE_SUBD_EVAL 0


float max_pixel_error = 5.0f; // maximal pixel error for DIFF_ERROR_IMAGE
float3 background_color = make_float3(1.f,1.f,1.f); // background color i.e. for teaser image should be white.
extern float aperture, focus_distance, eye_to_lens;


struct RayHitData{
	float2 TC;		 // texture coordinates.
	float3 bc, P, N; //	barycentric coordinates, hit position, hit normal
	material_t mat;  // current material
	rta::cuda::simple_triangle tri;	// triangle.
	float3 Tx, Ty;	// tangents Tx and Ty.
	float3 limitPosition; //limitPosition (in case of triangles = P)
	bool usePtexTexture;
};
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
// 	set.rgen = crgs = new rta::cuda::jittered_ray_generator(w, h, jitter);
	set.rgen = crgs = new rta::cuda::jittered_lens_ray_generator(w, h, focus_distance, aperture, eye_to_lens, jitter);
	int path_len = init_path_length;
	int paths = init_path_samples;
	set.bouncer = pt = new hybrid_pt_bouncer<B, T>(w, h, cpu_materials, triangles, crgs, cpu_lights, nr_of_lights, path_len, paths);
	
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
			if (curr_frame > 0)
				gi::save_image("ppt",curr_frame, w, h, colors);
			else
				gi::save_image("ppt", bouncer->curr_path, w, h, colors);
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

		// bouncer->path_samples = vars["pt/passes"].int_val;

		tracer->trace_progressively(true);
}
float3 evaluateSkyLight(gi::light *L, float3 &dir) {
	if (!L) return make_float3(0,0,0); //if not skylight return 0.

	float3 *skylightData = L->skylight.data;
	float theta = acosf(dir.y);
	float phi = atan2f(dir.z, dir.x);
	if (phi < 0) phi += 2.0f*float(M_PI);
	float s = phi/(2.0f*float(M_PI));
	float t = theta/float(M_PI);
	int idx = int(t*L->skylight.h) * L->skylight.w + int(s*L->skylight.w);
	//TODO: This should actually not happen :(
	if (idx < 0 || idx >= L->skylight.h * L->skylight.w) {return make_float3(0.f,0.f,0.f);}
 	return skylightData[idx];

}	
float clampFloat(float a) {
	if (a<0.0f) return 0.0f;
	if (a>1.0f) return 1.0f;
	return a;
}

void handle_no_hit(int id, float3 &orgDir,float3* ray_orig,float3 *ray_dir, float* max_t,float3* throughput,float3 *col_accum,gi::light *skylight,bool evalSkylight, float3 &backgroundColor){
	if (max_t[id] != -1 && evalSkylight) {
		col_accum[id] += throughput[id] * evaluateSkyLight(skylight,orgDir);
	}else{
		col_accum[id] += throughput[id] * backgroundColor;
	}
	ray_dir[id]  = make_float3(0,0,0);
	ray_orig[id] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	max_t[id] = -1;
	throughput[id] = make_float3(0,0,0);
}

float3 handle_error_image(int id, float3 *ray_dir, float3 *ray_orig, float3 &dummyP, float h){
	float3 rayDirection =ray_dir[id]; normalize_vec3f(&rayDirection);
	float3 correctP = dummyP;
	float3 correctDir = correctP-ray_orig[id];
	float3 pr = ray_orig[id] + (correctDir|rayDirection) * rayDirection;

	camera_ref cam = current_camera();
	matrix4x4f *mat = projection_matrix_of_cam(cam);
				
	vec4f correctP_vec4(correctP.x,correctP.y,correctP.z,1.0f);
	vec4f pr_vec4(pr.x,pr.y,pr.z,1.0f);
	//project to screen space
	vec4f correctP_proj;
	vec4f pr_proj;
	multiply_matrix4x4f_vec4f(&correctP_proj, mat, &correctP_vec4);
	multiply_matrix4x4f_vec4f(&pr_proj, mat, &pr_vec4);

	float2 aa = make_float2(correctP_proj.x,correctP_proj.y);
	aa.x = 0.5 * (aa.x * (1.0f/correctP_proj.w)) + 0.5;
	aa.y = 0.5 * (aa.y * (1.0f/correctP_proj.w)) + 0.5;
	float2 bb = make_float2(pr_proj.x,pr_proj.y);
	bb.x = 0.5 * ( bb.x * (1.0f/pr_proj.w)) + 0.5;
	bb.y = 0.5 * ( bb.y * (1.0f/pr_proj.w)) + 0.5;

	vec2f np;
	camera_near_plane_size(cam,&np);
					
	float2 ds3; 
	ds3.x = (aa.x-bb.x) * np.x;
	ds3.y = (aa.y-bb.y) * np.y;
	float dist_screen = sqrt(ds3.x*ds3.x+ds3.y*ds3.y);
	float pixel_size = np.y/float(h);

	float3 test;
	hsvColorMap(&test.x, dist_screen, 0.0f,max_pixel_error*pixel_size);
	return test;
}

float3 handle_pattern_image(float u, float v){
	float3 whiteC = make_float3(0.8f,0.8f,0.8f);
	float3 blackC = make_float3(0.2f,0.2f,0.2f);
	float3 color = whiteC;
	//int uInt = int(u*100.0f);
	//int vInt = int(v*100.0f);
float uInt = u;
float vInt = v;	
	
	if(uInt >= 0.45f && uInt <= 0.55f) return blackC;
	if(vInt >= 0.45f && vInt <= 0.55f) return blackC;

	if(uInt >= 0.25f && uInt <= 0.3f) return blackC;
	if(vInt >= 0.25f && vInt <= 0.3f) return blackC;

	if(uInt >= 0.7f && uInt <= 0.75f) return blackC;
	if(vInt >= 0.7f && vInt <= 0.75f) return blackC;

	//if(vInt >= 0.1f && vInt <= 0.15f) return blackC;
//	if(vInt >= 0.85 && vInt <=0.9) return blackC;
	return whiteC;
}

//Time for evaluating FAS.
void time_adaptive_subd(int w, int h,triangle_intersection<rta::cuda::simple_triangle> *ti, bool with_disp){
	wall_time_t start = wall_time_in_ms();
	#pragma omp parallel for
	for(int y=0; y<h; ++y){
		for(int x =0; x<w; ++x){
		int2 gid = make_int2(x, y);
		int id = gid.y*w+gid.x;
		triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
		if(is.valid()){
			if ((is.ref & 0xff000000) == 0) {
				//triangel => continue
				continue;
			}
			#if HAVE_LIBOSDINTERFACE == 1
			unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
			unsigned int ptexID = 0x00ffffff & is.ref;
			float3 dummyP;
			float3 dummyN;
			is.beta = clampFloat(is.beta);
			is.gamma = clampFloat(is.gamma);
			subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, with_disp, (float*)&dummyP, (float*)&dummyN);
			#endif
			}
		}
	}
	wall_time_t stop = wall_time_in_ms();
	time_adaptive_subd_eval += stop - start;
	wall_time_t startAll = wall_time_in_ms();
}

void computeRayDiffs(int id, int w, int h,rta::cuda::simple_triangle &tri,float3 bc, float2 TC, float3 &N, float3 *ray_diff_org,float3 *ray_diff_dir, float2& upper_T, float2& right_T){
	float3 upper_org = ray_diff_org[id];
	// eval ray differentials (stored below)
	float3 upper_dir = ray_diff_dir[id];
    float3 right_org = ray_diff_org[w*h+id];
	float3 right_dir = ray_diff_dir[w*h+id];
	float3 upper_P, right_P;
//	float2 upper_T, right_T;
	/* temporarily disabled ray differentials. (had to enable it to get correct texture lookup for .obj) 
	 * 	Undefined behavior in case of Subd Models!
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
	ray_diff_dir[id] = reflect(upper_dir, N);
}

bool sample_material(int id, float3 &T, float3 &B, float3 &N, float3 *uniform_random, float3 &org_dir, float3 &inv_org_dir,materialBRDF &currentMaterial, float &pdf, float3 &sampledDirection,bool &enterGlass){
	float3 random = next_random3f(uniform_random, id);
	pdf = 1.f;
	enterGlass = true;
	if ((org_dir|N) > 0.0f) {
		if (currentMaterial.isGlass()) {
			enterGlass = false;
			N *= -1.f;
		}else{
			N *= -1.f;
//			return false;
		}
	}
	float3 inv_org_dir_ts = transform_to_tangent_frame(inv_org_dir,T,B,N);
	currentMaterial.sample(inv_org_dir_ts,sampledDirection, random,pdf, enterGlass);
	sampledDirection = transform_from_tangent_frame(sampledDirection,T,B,N);
	normalize_vec3f(&sampledDirection);
	return true;

}

float3 evaluate_material(int id, float3 *throughput,float3 *to_light, float3* potential_sample_contribution,float3 &N, float3* col_accum, 
	float3 &inv_org_dir,materialBRDF &currentMaterial,triangle_intersection<rta::cuda::simple_triangle> *shadow_ti, float3 &sampledDirection){
	//do shading for non-glass materials.
	if (!currentMaterial.isGlass()){// && !currentMaterial.specReflect()) {
		triangle_intersection<rta::cuda::simple_triangle> is = shadow_ti[id];
		if (!is.valid()) {
			float3 weight = potential_sample_contribution[id];
			// attention: we need the throughput *before* the bounce
			float3 currTP = throughput[id] * weight;
			float3 light_dir = to_light[id];
			normalize_vec3f(&light_dir);
			// the whole geometric term is already computed in potential_sample_contribution.
			float3 brdfLight = currentMaterial.evaluate(inv_org_dir,light_dir,N);
			col_accum[id] +=  brdfLight * currTP;
		}
	}
	// compute next path segment by sampling the brdf
	float3 brdf = currentMaterial.evaluate(inv_org_dir,sampledDirection,N); //throughput is multiplied onto brdf later on.
	return brdf;

}

#if HAVE_LIBOSDINTERFACE == 1
void getHitDataSubdSurface(triangle_intersection<rta::cuda::simple_triangle>& is,rta::cuda::material_t *mats, RayHitData &hdata){

// evaluate subd patch to get position and normal
	unsigned int modelidx = (0x7f000000 & is.ref) >> 24;
	unsigned int ptexID = 0x00ffffff & is.ref;
	//bool WITH_DISPLACEMENT = true;//false;//true;//false;// true;//false;
	float mipmapBias = 0.f;
	is.beta = clampFloat(is.beta);
	is.gamma = clampFloat(is.gamma);
	bool withDip = true;
	//!TODO:tangents are only written IF no displacement? Why?
	if (withDip)
		subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, true, (float*)&hdata.limitPosition, (float*)&hdata.N);
	else
		subd_models[modelidx]->EvalLimit(ptexID, is.beta, is.gamma, false, (float*)&hdata.limitPosition, (float*)&hdata.N);//, mipmapBias, (float*)&Tx, (float*)&Ty);
	// evaluate color and store it in the material as diffuse component
	int materialIndex = idx_subd_material  - 1 +  subd_models[modelidx]->GetMaterialIndex() ;//+ idx_subd_material;
	if (materialIndex < 0  || materialIndex>= material_count) {
		std::cerr << "Warning: Material index " << materialIndex <<" build from "<<subd_models[modelidx]->GetMaterialIndex()<< "  is out of bounds " << material_count << "\n";
		materialIndex = material_count - 1; // Set to default material.
	}
	hdata.mat = mats[materialIndex];
	hdata.usePtexTexture = false;//true;
		
	//clamp uvs (not necessary)
	#if RENDER_UVS || DIFF_ERROR_IMAGE || RENDER_PATTERN
		//do nothing
		hdata.mat.diffuse_color = make_float3(1.f,0.f,1.f);
	#else
	if(subd_models[modelidx]->HasColor()){
		hdata.usePtexTexture = true;
		subd_models[modelidx]->EvalColor(ptexID, is.beta, is.gamma, (float*)&hdata.mat.diffuse_color);					
	}else{
		if(hdata.mat.isPrincipledMaterial())	
			hdata.mat.diffuse_color = hdata.mat.parameters->color;
	}
	#endif
	//!TODO Setup "correct" triangle for ray differentials, is this neccessary?
	hdata.P = hdata.limitPosition;
	hdata.tri.a = hdata.P; hdata.tri.b = hdata.P+hdata.Tx; hdata.tri.c = hdata.P+hdata.Ty;
	normalize_vec3f(&hdata.N);
	hdata.TC.x = is.beta;
	hdata.TC.y = is.gamma;

}
#endif

void getHitDataTriangle(triangle_intersection<rta::cuda::simple_triangle>& is, rta::cuda::simple_triangle *triangles, rta::cuda::material_t *mats, RayHitData &hdata){
	// load hit triangle and compute hitpoint geometry
	// we load the material, too, as it is stored in the triangle
	hdata.tri = triangles[is.ref];
	hdata.mat = mats[hdata.tri.material_index];
	is.barycentric_coord(&hdata.bc);
	barycentric_interpolation(&hdata.P, &hdata.bc, &hdata.tri.a, &hdata.tri.b, &hdata.tri.c);
	barycentric_interpolation(&hdata.N, &hdata.bc, &hdata.tri.na, &hdata.tri.nb, &hdata.tri.nc);
	barycentric_interpolation(&hdata.TC,&hdata.bc, &hdata.tri.ta, &hdata.tri.tb, &hdata.tri.tc);
	normalize_vec3f(&hdata.N);

	float3 dpos = hdata.tri.b - hdata.tri.a;
	float3 triTb3 = make_float3(hdata.tri.tb.x,hdata.tri.tb.y,1.0f);
	float3 triTa3 = make_float3(hdata.tri.ta.x,hdata.tri.ta.y,1.0f);
	float3 duv3 = triTb3 - triTa3;
	if (duv3.x == 0) {
		dpos = hdata.tri.c - hdata.tri.a;
		float3 triTb3 = make_float3(hdata.tri.tc.x,hdata.tri.tc.y,1.0f);
		duv3 = hdata.tri.c - hdata.tri.a;
		if (duv3.x == 0) duv3.x = 1.0f;
	}
	hdata.Tx = dpos * (1.0f/duv3.x);
	normalize_vec3f(&hdata.Tx);
	cross_vec3f(&hdata.Ty,&hdata.Tx,&hdata.N);
	normalize_vec3f(&hdata.Ty);
	hdata.limitPosition = hdata.P;

}

void compute_path_contribution_and_bounce(int w, int h, float3 *ray_orig, float3 *ray_dir, float *max_t, float3 *ray_diff_org, float3 *ray_diff_dir,
										  triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles, 
										  rta::cuda::material_t *mats, float3 *uniform_random, float3 *throughput, float3 *col_accum,
										  float3 *to_light, triangle_intersection<rta::cuda::simple_triangle> *shadow_ti,
										  float3 *potential_sample_contribution, gi::light *skylight, int pathLen) {


	bool WITH_DISPLACEMENT = true;
#if TIME_ADAPTIVE_SUBD_EVAL
	time_adaptive_subd(w,h,ti,WITH_DISPLACEMENT);
#endif
	
	#pragma omp parallel for 
	for (int y = 0; y < h; ++y) {
		materialBRDF currentMaterial;
		for (int x = 0; x < w; ++x) {
			// general setup, early out if the path intersection is invalid.
			int2 gid = make_int2(x, y);
			int id = gid.y*w+gid.x;
			triangle_intersection<rta::cuda::simple_triangle> is = ti[id];
			//ray direction (normalized).
			float3 org_dir = ray_dir[id];
			normalize_vec3f(&org_dir);

			if (is.valid()) {
				RayHitData hdata;
				// check if we hit a triangle or a subd patch
				if ((is.ref & 0xff000000) == 0) {
					getHitDataTriangle(is,triangles,mats,hdata);
				} else {
#if HAVE_LIBOSDINTERFACE == 1
					getHitDataSubdSurface(is, mats,hdata);
					hdata.P = ray_orig[id] + (is.t) * org_dir;//ray_dir[id];
#endif
				}
				bool isPrincipled = hdata.mat.isPrincipledMaterial();
				/****************************************************************************************************************************/
				// HANDLE DIFFERENT RENDERING SETUPS : i.e. render uvs, render error image
				/****************************************************************************************************************************/
				#if RENDER_UVS
					hdata.mat.diffuse_color = make_float3(hdata.TC.x, hdata.TC.y, 0.0f);
					handle_no_hit(id,org_dir,ray_orig,ray_dir,max_t,throughput,col_accum,skylight,false,hdata.mat.diffuse_color);
					isPrincipled = false;
					continue;
				#elif DIFF_ERROR_IMAGE
					float3 errorColor = handle_error_image(id, ray_dir, ray_orig,hdata.limitPosition,h);
					handle_no_hit(id,org_dir,ray_orig,ray_dir,max_t,throughput,col_accum,skylight,false,errorColor);
					isPrincipled = false;
					continue;
				#elif RENDER_PATTERN
					float3 col = make_float3(0.0f,0.0f,0.0f);
					if ((is.ref & 0xff000000) != 0) {
						unsigned int ptexID = 0x00ffffff & is.ref;
						if(ptexID == 0) col.x = 1.f;
						else if(ptexID==1) col.y = 1.f;
						else col.z = 1.f;
					}
					float3 pattern = handle_pattern_image(hdata.TC.x,hdata.TC.y) * col;//*make_float3(hdata.TC.x, hdata.TC.y, 0.0f);
					hdata.mat.diffuse_color = pattern;
					hdata.usePtexTexture = true;//false;
					isPrincipled = false;
					handle_no_hit(id,org_dir,ray_orig,ray_dir,max_t,throughput,col_accum,skylight,false,pattern);
					continue;
				#endif
				/****************************************************************************************************************************/


				// Compute Ray differentials.
				float2 upper_T, right_T;
				computeRayDiffs(id, w, h,hdata.tri,hdata.bc, hdata.TC,hdata.N, ray_diff_org,ray_diff_dir,upper_T, right_T);

				/****************************************************************************************************************************/
				// MATERIAL HANDLING
				/****************************************************************************************************************************/
				//construct Tangent Frame.
				float3 T, B;				
				make_tangent_frame(hdata.N, T, B);
				normalize_vec3f(&T);
				normalize_vec3f(&B);
				// load Material.
				currentMaterial.init(isPrincipled,hdata.usePtexTexture,&hdata.mat, hdata.TC, upper_T, right_T, T, B);

				// sample Material.	
				float pdf = 1.f;
				float3 inv_org_dir = -1.0f*org_dir;
				float3 sampledDirection;
				bool enterGlass = true;
				if(!sample_material(id,T,B, hdata.N, uniform_random, org_dir, inv_org_dir,currentMaterial,pdf,sampledDirection,enterGlass)){
					handle_no_hit(id,org_dir,ray_orig,ray_dir,max_t,throughput,col_accum,skylight,false,background_color);
					continue;
				}

				// evaluate Material.
				float3 TP = throughput[id];
				float3 brdf = evaluate_material(id,throughput,to_light, potential_sample_contribution,hdata.N, 
												col_accum, inv_org_dir,currentMaterial,shadow_ti, sampledDirection);

				/****************************************************************************************************************************/
				// SETUP NEXT RAY
				/****************************************************************************************************************************/
				// offset position.
				float offsetFactor = 0.3f;// Offset for car model : 0.001f, offset for Trex: 0.3f
				if(currentMaterial.isGlass())
					hdata.P += offsetFactor * org_dir;
				else
					hdata.P += offsetFactor * hdata.N;
				ray_orig[id] = hdata.P;
				ray_dir[id]  = sampledDirection;
				max_t[id]    = FLT_MAX;
				throughput[id] = TP *  brdf/pdf;
				if(throughput[id].x > 1) throughput[id].x = 1.f;
				if(throughput[id].y > 1.f) throughput[id].y = 1.f;
				if(throughput[id].z > 1.f) throughput[id].z = 1.f;
				continue;

			}//end isValid(is);

#ifdef NO_BACKGROUND
			// Do NOT render any skylight but the color defined in background_color.
			if (throughput[id].x == 1.0f && throughput[id].y == 1.0f && throughput[id].z == 1.0f) { //check wether this is the initial pixel ray (probably also possible with depth == 0?)
				handle_no_hit(id,org_dir,ray_orig,ray_dir,max_t,throughput,col_accum,skylight,false,background_color);
				continue;
			}
#endif
			handle_no_hit(id, org_dir,ray_orig, ray_dir, max_t, throughput,col_accum,skylight,true,background_color);
			continue;
		}
	}

#if TIME_ADAPTIVE_SUBD_EVAL
	wall_time_t endAll = wall_time_in_ms();
	double allT = endAll-startAll;
	std::cerr<<"Time path_contrib_bounce: "<<allT<<"ms, of which  subd eval was "<<time_adaptive_subd_eval<<" ms \n";
#endif
}


/* vim: set foldmethod=i*/

