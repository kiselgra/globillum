#include "config.h"
#include "cpu-pt.h"

#include "raygen.h"
#include "material-wrapper.h"

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
#if HAVE_LIBOSDINTERFACE == 1
extern std::vector<OSDI::Model*> subd_models;
#endif

#define DEBUG_PBRDF 0

extern float aperture, focus_distance, eye_to_lens;

void cpu_pt::activate(rt_set *orig_set) {
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
	
	rta::basic_acceleration_structure<B,T> *bas = set.basic_as<B, T>();
	triangles = bas->canonical_triangle_ptr();

// 	set.rgen = crgs = new rta::cuda::camera_ray_generator_shirley<rta::cuda::gpu_ray_generator_with_differentials>(w, h);
	jitter = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	update_mt_pool(jitter);
	cpu_jitter = new float3[w*h];
	set.rgen = crgs = new rta::jittered_lens_ray_generator(w, h, focus_distance, aperture, eye_to_lens, cpu_jitter);
	int bounces = 1;
	set.bouncer = pt = new cpu_pt_bouncer<B, T>(w, h, cpu_materials, triangles, crgs, cpu_lights, nr_of_lights, bounces, vars["pt/passes"].int_val);
	
	gi::cuda::mt_pool3f pl = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	gi::cuda::mt_pool3f pp = gi::cuda::generate_mt_pool_on_gpu(w,h); 
	update_mt_pool(pl);
	update_mt_pool(pp);
	pt->random_number_generator(pl, pp);


	// setup iterated tracers
	rta::cpu_raytracer<B, T, rta::closest_hit_tracer> 
		*cpu_tracer = dynamic_cast<rta::cpu_raytracer<B, T, rta::closest_hit_tracer>*>(set.rt);
	cpu_tracer->ray_bouncer(set.bouncer);
	cpu_tracer->ray_generator(set.rgen);
	tracers = new rta::iterated_cpu_tracers<B, T, rta::closest_hit_tracer>(cpu_tracer);

	shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	rta::cpu_raytracer<B, T, rta::any_hit_tracer> 
		*shadow_cpu_tracer = dynamic_cast<rta::cpu_raytracer<B, T, rta::any_hit_tracer>*>(shadow_tracer);
	shadow_cpu_tracer->ray_bouncer(set.bouncer);
	shadow_cpu_tracer->ray_generator(set.rgen);
	shadow_tracers = new rta::iterated_cpu_tracers<B, T, rta::any_hit_tracer>(shadow_cpu_tracer);

	if (original_subd_set && false) { // this is not supported yet.
		// subd closest hit
		rta::cpu_raytracer<B, T, rta::closest_hit_tracer>
			*subd_tracer = dynamic_cast<rta::cpu_raytracer<B, T, rta::closest_hit_tracer>*>(original_subd_set->rt);
		subd_tracer->ray_generator(set.rgen);
		subd_tracer->ray_bouncer(set.bouncer);
		tracers->append_tracer(subd_tracer);
		// subd any hit
		rta::cpu_raytracer<B, T, rta::any_hit_tracer>
			*subd_shadow_tracer = dynamic_cast<rta::cpu_raytracer<B, T, rta::any_hit_tracer>*>(subd_tracer->matching_any_hit_tracer());
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

bool cpu_pt::in_progress() {
	return (tracer && tracer->progressive_trace_running());
}

void cpu_pt::update() {
	if (tracer->progressive_trace_running()) {
		tracer->trace_progressively(false);
		cpu_pt_bouncer<B,T> *bouncer = dynamic_cast<cpu_pt_bouncer<B, T>*>(set.bouncer);
		float3 *colors = bouncer->output_color;
		if (bouncer->path_len == 0) {
			update_mt_pool(jitter);	// a path is completed, so we generate new random numbers for the primary ray generator.
			gi::save_image("pt", bouncer->curr_path, w, h, colors);
		}
	}
}

void cpu_pt::compute() {
		cout << "restarting progressive display" << endl;
		vec3f pos, dir, up;
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
		extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
		extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
		extract_up_vec3f_of_matrix(&up, lookat_matrix);
		if (nr_of_lights != gi::lights.size())
			throw std::runtime_error("number of lights changed in " "cpu_pt::compute" 
									 " this exception is just for consistency to the gpu version.");
		overall_light_power = 0;
		for (int i = 0; i < gi::lights.size(); ++i) {
			cpu_lights[i] = gi::lights[i];
			overall_light_power += cpu_lights[i].power;
		}
		cpu_pt_bouncer<B,T> *bouncer = dynamic_cast<cpu_pt_bouncer<B, T>*>(set.bouncer);
		bouncer->overall_light_power = overall_light_power;
		bouncer->verbose = verbose;
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

		bouncer->path_samples = vars["pt/passes"].int_val;

		tracer->trace_progressively(true);
}




/* vim: set foldmethod=marker: */

