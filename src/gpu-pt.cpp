#include "gpu-pt.h"



using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;


void gpu_pt::activate(rt_set *orig_set) {
// 	gpu_pt_bouncer<B, T>::random_number_generator_t rng_t = gpu_pt_bouncer<B, T>::simple_halton;
// 	gpu_pt_bouncer<B, T>::random_number_generator_t rng_t = gpu_pt_bouncer<B, T>::lcg;
	gpu_pt_bouncer<B, T>::random_number_generator_t rng_t = gpu_pt_bouncer<B, T>::per_frame_mt;
	if (activated) return;
	declare_variable<int>("pt/passes", 32);
	gi_algorithm::activate(orig_set);
	set = *orig_set;
	set.rt = set.rt->copy();
	gpu_rect_lights = cuda::cgls::convert_and_upload_rectangular_area_lights(scene, nr_of_gpu_rect_lights);
	cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
	set.rgen = crgs = new cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(w, h);
	int bounces = 4;
	set.bouncer = pt = new gpu_pt_bouncer<B, T>(w, h, gpu_materials, triangles, crgs, gpu_rect_lights, nr_of_gpu_rect_lights, bounces, vars["pt/passes"].int_val);
	gi::halton_pool2f halton_pool;
	gi::lcg_random_state lcg_pool;
	if (rng_t == gpu_pt_bouncer<B, T>::simple_halton)
		pt->random_number_generator(gi::cuda::generate_halton_pool_on_gpu(w*h));
	else if (rng_t == gpu_pt_bouncer<B, T>::lcg)
		pt->random_number_generator(gi::cuda::generate_lcg_pool_on_gpu(w*h));
	else if (rng_t == gpu_pt_bouncer<B, T>::per_frame_mt) {
		gi::cuda::mt_pool3f pl = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		gi::cuda::mt_pool3f pp = gi::cuda::generate_mt_pool_on_gpu(w,h); 
		update_mt_pool(pl);
		update_mt_pool(pp);
		pt->random_number_generator(pl, pp);
	}
	set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
	set.basic_rt<B, T>()->ray_generator(set.rgen);
	shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
	tracer = new tandem_tracer<B, T>(dynamic_cast<basic_raytracer<B,T>*>(set.rt), 
									 dynamic_cast<basic_raytracer<B,T>*>(shadow_tracer));
	tracer->select_closest_hit_tracer();
	pt->register_tracers(tracer);

	rta::cuda::cgls::init_cuda_image_transfer(result);
}

void gpu_pt::update() {
	if (tracer->progressive_trace_running()) {
		tracer->trace_progressively(false);
		gpu_pt_bouncer<B,T> *bouncer = dynamic_cast<gpu_pt_bouncer<B, T>*>(set.bouncer);
		float3 *colors = bouncer->output_color;
// 		float3 *colors = bouncer->material_colors;
		cuda::cgls::copy_cuda_image_to_texture(w, h, colors, 1.0f);
	}
}

void gpu_pt::compute() {
		cout << "restarting progressive display" << endl;
		vec3f pos, dir, up;
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
		extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
		extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
		extract_up_vec3f_of_matrix(&up, lookat_matrix);
		rta::cuda::cgls::update_rectangular_area_lights(scene, gpu_rect_lights, nr_of_gpu_rect_lights);
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

		gpu_pt_bouncer<B,T> *bouncer = dynamic_cast<gpu_pt_bouncer<B, T>*>(set.bouncer);
		bouncer->path_samples = vars["pt/passes"].int_val;

		tracer->trace_progressively(true);
}
	
/* vim: set foldmethod=marker: */

