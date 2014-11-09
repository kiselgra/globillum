#include "gpu-pt.h"



using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;

void gpu_pt::activate(rt_set *orig_set) {
	set = *orig_set;
	set.rt = set.rt->copy();
	gpu_rect_lights = cuda::cgls::convert_and_upload_rectangular_area_lights(scene, nr_of_gpu_rect_lights);
	cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
	set.rgen = crgs = new cuda::cam_ray_generator_shirley(w, h);
	gi::cuda::halton_pool2f pool = gi::cuda::generate_halton_pool_on_gpu(w*h);
	set.bouncer = new gpu_pt_bouncer<B, T>(w, h, gpu_materials, triangles, crgs, gpu_rect_lights, nr_of_gpu_rect_lights, pool, 1);
	set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
	set.basic_rt<B, T>()->ray_generator(set.rgen);

	cuda::cgls::init_cuda_image_transfer(result);
}

void gpu_pt::update() {
	if (shadow_tracer && shadow_tracer->progressive_trace_running()) {
		cout << "t" << endl;
		shadow_tracer->trace_progressively(false);
		gpu_pt_bouncer<B,T> *bouncer = dynamic_cast<gpu_pt_bouncer<B, T>*>(set.bouncer);
		float3 *colors = bouncer->output_color;
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
		update_rectangular_area_lights(scene, gpu_rect_lights, nr_of_gpu_rect_lights);
		crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));

		set.rt->trace_progressively(true);
		shadow_tracer = dynamic_cast<rta::closest_hit_tracer*>(set.rt)->matching_any_hit_tracer();
}
	
/* vim: set foldmethod=marker: */

