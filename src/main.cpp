#define PNG_SKIP_SETJMP_CHECK
#include <libcgls/cgls.h>
#include <libcgls/picking.h>
#include <libcgls/interaction.h>
#include <libcgls/console.h>
#include <libcgl/wall-time.h>

#ifndef SCM_MAGIC_SNARFER

#include <librta/material.h>

#include <libhyb/rta-cgls-connection.h>

#include "cmdline.h"
#include "noscm.h"

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"
#include "gpu-pt.h"

#include "material.h"
#include "vars.h"
#include "rayvis.h"
#include "subd.h"

#include <GL/freeglut.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

scene_ref the_scene;
#define time_measurement_samples 128
float times[time_measurement_samples];
int valid_pos = 0, curr_pos = 0;

extern int subd_tess_normal;
extern int subd_tess_quant;
extern float subd_disp_scale;
extern std::string subd_face_include;

framebuffer_ref gbuffer;
picking_buffer_ref picking;

console_ref viconsole = { -1 };
bool gb_debug = false;

std::map<std::string, var> vars;

mesh_ref quad;
shader_ref quad_shader;
bool show_results = false;
float exposure = 10;

// used for dof implementations
float aperture = .5;
float focus_distance = 970.0f;
float eye_to_lens = 5.0f;
int idx_subd_material = 0;

void display() {
	
	if (cgl_shader_reload_pending)
		reload_shaders();
	reload_modified_shader_files();
	
	if (shader_errors_present()) {
		render_shader_error_message();
		swap_buffers();
		return;
	}
		
	glDisable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEPTH_TEST);

// 	restart_rayvis();
// 	add_vertex_to_all_rays(make_float3(0,0,0));
// 	add_vertex_to_all_rays(make_float3(0,100,0));
// 	add_vertex_to_all_rays(make_float3(1000,100,0));
// 	add_vertex_to_all_rays(make_float3(1000,400,0));
	
    glFinish();
	wall_time_t start = wall_time_in_ms();

// 	glClearColor(0,0,0.25,1);
// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
// 	render_scene(the_scene);
//     unbind_framebuffer(gbuffer);
// 
// 	render_scene_deferred(the_scene, gbuffer);

	if (gi_algorithm::selected && gi_algorithm::selected->progressive())
		gi_algorithm::selected->update();

	if (show_results) {
		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		bind_shader(quad_shader);
		bind_mesh_to_gl(quad);
		texture_ref tex = gi_algorithm::result;
// 		tex = find_texture("gbuffer/diffuse");
		bind_texture(tex, 0);
		glUniform1f(0, pow(2.0, (exposure-10.0)));
		draw_mesh(quad);
		unbind_texture(tex);
		unbind_mesh_from_gl(quad);
		unbind_shader(quad_shader);
	}
	else {
		bind_framebuffer(gbuffer);
		glClearColor(0,0,0.25,1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		unbind_framebuffer(gbuffer);

		render_scene_to_gbuffer(the_scene, gbuffer);
		if (!gb_debug)
			render_scene_from_gbuffer(the_scene, gbuffer);
		else
			render_gbuffer_visualization(the_scene, gbuffer);

// 		glClear(GL_DEPTH_BUFFER_BIT);
// 		render_rayvis();
	}

	glFinish();
	wall_time_t end = wall_time_in_ms();

	times[curr_pos] = end-start;
	curr_pos = (curr_pos+1) % time_measurement_samples;
	valid_pos = (valid_pos == time_measurement_samples ? time_measurement_samples : valid_pos+1);

    check_for_gl_errors("end of display");

	render_console(viconsole);
	swap_buffers();
}

void idle() {
	glutPostRedisplay(); 
}

void gb_dbg(interaction_mode *m, int x, int y) {
	gb_debug = !gb_debug;
}

void showres(interaction_mode *m, int x, int y) {
	show_results = !show_results;
}

void show_fps(interaction_mode *m, int x, int y) {
	double sum = 0;
	for (int i = 0; i < valid_pos; ++i)
		sum += times[i];
	float avg = sum / (double)valid_pos;
	printf("average render time: %.3f ms, %.1f fps \t(sum %f, n %d)\n", avg, 1000.0f/avg, (float)sum, valid_pos);
}

void compute_trace(interaction_mode *m, int x, int y) {
	gi_algorithm::selected->compute();
}

static rta::cgls::connection *rta_connection = 0;
rta::basic_flat_triangle_list<rta::simple_triangle> *ftl = 0;
rta::cgls::connection::cuda_triangle_data *ctd = 0;
int material_count = 0;
rta::cuda::material_t *gpu_materials = 0;
rta::cuda::material_t *cpu_materials = 0;


void setup_rta(std::string plugin) {
	bool use_cuda = true;
	vector<string> args;
	int rays_w = cmdline.res.x, rays_h = cmdline.res.y;

#if HAVE_LIBOSDINTERFACE == 1
	// load subd plugin
	rta::rt_set *subd_set = generate_compressed_bvhs_and_tracer(rays_w, rays_h);
	gi_algorithm::original_subd_set = subd_set;
#endif
		
	if (plugin == "default/choice")
		if (use_cuda) 
			plugin = "bbvh-cuda";
		else
			plugin = "bbvh";
			
	if (plugin == "bbvh-cuda") {
		args.push_back("-A");
		args.push_back("-b");
// 		args.push_back("bsah");
		args.push_back("median");
		args.push_back("-t");
		args.push_back("cis");
		args.push_back("-l");
		args.push_back("2f4");
	}

	rta_connection = new rta::cgls::connection(plugin, args);
	ctd = rta::cgls::connection::convert_scene_to_cuda_triangle_data(the_scene);
	static rta::basic_flat_triangle_list<rta::simple_triangle> the_ftl = ctd->cpu_ftl();
	ftl = &the_ftl;
	rta::rt_set *set = new rta::rt_set(rta::plugin_create_rt_set(*ftl, rays_w, rays_h));

	std::vector<std::string> dummyF(0);
	gpu_materials = rta::cuda::convert_and_upload_materials(material_count,dummyF);
	cpu_materials = rta::cuda::download_materials(gpu_materials, material_count);

	if (!use_cuda) {
// 		use_case = new example::simple_lighting_with_shadows<rta::simple_aabb, rta::simple_triangle>(set, rays_w, rays_h, the_scene);
	}
	else {
		if (!set->basic_ctor<rta::cuda::simple_aabb, rta::cuda::simple_triangle>()->expects_host_triangles()) {
			cout << "does not want host tris!" << endl;
			typedef rta::cuda::simple_aabb box_t;
			typedef rta::cuda::simple_triangle tri_t;
			set->as = set->basic_ctor<box_t,tri_t>()->build((rta::cuda::simple_triangle::input_flat_triangle_list_t*)&ctd->ftl);
			set->basic_rt<box_t,tri_t>()->acceleration_structure(dynamic_cast<rta::basic_acceleration_structure<box_t,tri_t>*>(set->as));
			cout << "done!" << endl;
		}
// 		use_case = new example::simple_lighting_with_shadows<rta::cuda::simple_aabb, rta::cuda::simple_triangle>(set, rays_w, rays_h, the_scene);
	}

	tex_params_t p = default_fbo_tex_params();
	gi_algorithm::result = make_empty_texture("gi-res", cmdline.res.x, cmdline.res.y, GL_TEXTURE_2D, GL_RGBA32F, GL_FLOAT, GL_RGBA, &p);
	gi_algorithm::original_rt_set = set;

	// we load the subd proxies *after* the ftl has been built.
	load_subd_proxies();
}


interaction_mode* make_viewer_mode() {
	interaction_mode *m = make_interaction_mode("viewer");
	add_function_key_to_mode(m, 'p', cgls_interaction_no_button, show_fps);
	add_function_key_to_mode(m, ' ', cgls_interaction_no_button, compute_trace);
	add_function_key_to_mode(m, '~', cgls_interaction_no_button, gb_dbg);
	add_function_key_to_mode(m, 'R', cgls_interaction_shift, showres);
	return m;
}

void adjust_view(const vec3f *bb_min, const vec3f *bb_max, vec3f *cam_pos, float *distance) {
	vec3f bb_center, tmp;
	sub_components_vec3f(&tmp, bb_max, bb_min);
	div_vec3f_by_scalar(&tmp, &tmp, 2);
	add_components_vec3f(&bb_center, &tmp, bb_min);
	
	sub_components_vec3f(&tmp, bb_max, bb_min);
	*distance = length_of_vec3f(&tmp);
	make_vec3f(&tmp, 0, 0, *distance);
	add_components_vec3f(cam_pos, &bb_center, &tmp);

	cgl_cam_move_factor = *distance / 20.0f;
}

#ifdef WITH_GUILE
extern "C" {
void register_scheme_functions_for_cmdline();
void register_scheme_functions_for_light_setup();
static void register_scheme_functions_for_main();
}
static void register_scheme_functions() {
	register_scheme_functions_for_cmdline();
	register_scheme_functions_for_light_setup();
	register_scheme_functions_for_main();
}
#endif

// console stuff

static char* console_screenshot(console_ref ref, int argc, char **argv) {
// 	if (argc != 2)
// 		return strdup("requries a single argument.");
// 	screenshot_name = argv[1];
// 	take_screenshot_in_next_frame = true;
	return 0;
}

static char* console_bookmark(console_ref ref, int argc, char **argv) {
	if (argc != 2)
		return strdup("name required");
	vec3f cam_pos, cam_dir, cam_up;
	matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
	extract_pos_vec3f_of_matrix(&cam_pos, lookat_matrix);
	extract_dir_vec3f_of_matrix(&cam_dir, lookat_matrix);
	extract_up_vec3f_of_matrix(&cam_up, lookat_matrix);
	ofstream out("bookmarks", fstream::out|fstream::app);
	char *base = basename((char*)cmdline.filename);
	out << "(bookmark \"" << base << "\" \"" << argv[1];
	out << "\" '((list " << cam_pos.x << " " << cam_pos.y << " " << cam_pos.z << ")";
	out << "     (list " << cam_dir.x << " " << cam_dir.y << " " << cam_dir.z << ")";
	out << "     (list " << cam_up.x  << " " << cam_up.y  << " " << cam_up.z << ")))" << endl;
	scm_c_eval_string("(set! bookmarks '())");
	scm_c_eval_string("(primitive-load \"bookmarks\")");
	return 0;
}

static char* console_select_bookmark(console_ref ref, int argc, char **argv) {
	if (argc != 2)
		return strdup("which one?");
	char *base = basename((char*)cmdline.filename);
	string expr = string("(select-bookmark \"") + base + "\" \"" + argv[1] + "\")";
	SCM r = scm_c_eval_string(expr.c_str());
	if (scm_is_false(r))
		return strdup("not found");
	return scm_to_locale_string(r);
}

static char* console_algo(console_ref ref, int argc, char **argv) {
	if (argc != 2)
		return strdup("requires a simple argument.");

	list<string> names = gi_algorithm::list();
	string found = "";
	for (string name : names) {
		if (name.find(argv[1]) != string::npos) {
			found = name;
			break;
		}
	}
	if (found != "") {
		gi_algorithm::select(found);
		return strdup(found.c_str());
	}
	else return strdup("not found");
}

static char* console_decl(console_ref ref, int argc, char **argv) {
	if (argc != 3)
		return strdup("requires variable name and type");
	string t = argv[2];
	var v;
	v.name = argv[1];
	if (t == "int") v.type = var::t_int;
	else if (t == "float") v.type = var::t_float;
	else return strdup(("unrecognized type " + t).c_str());
	vars[v.name] = v;
	return 0;
}

static char* console_set(console_ref ref, int argc, char **argv) {
	if (argc != 3)
		return strdup("requires variable name and new value");
	map<string, var>::iterator v = vars.find(argv[1]);
	if (v == vars.end())
		return strdup(("undeclared variable " + string(argv[1])).c_str());
	if (v->second.type == var::t_int)
		v->second.int_val = atoi(argv[2]);
	else
		v->second.float_val = atof(argv[2]);
	return 0;
}

static char* console_show(console_ref ref, int argc, char **argv) {
	if (argc != 2)
		return strdup("requires variable name");
	map<string, var>::iterator v = vars.find(argv[1]);
	if (v == vars.end())
		return strdup(("undeclared variable " + string(argv[1])).c_str());
	ostringstream oss;
	if (v->second.type == var::t_int)
		oss << v->second.int_val;
	else
		oss << v->second.float_val;
	return strdup(oss.str().c_str());
}

static char* console_exposure(console_ref ref, int argc, char **argv) {
	if (argc > 2)
		return strdup("can be called with up to one argument, sets exposure for hdr display.");
	if (argc > 1) {
		exposure = atof(argv[1]);
		return 0;
	}
	ostringstream oss;
	oss << "exposure = " << exposure;
	return strdup(oss.str().c_str());
}

static char* console_focus_distance(console_ref ref, int argc, char **argv) {
	if (argc > 2)
		return strdup("return focus distance when called without parameter, with one parameter focus distance ist set.");
	if (argc == 1) {
		ostringstream oss; oss << "focus distance = " << focus_distance;
		return strdup(oss.str().c_str());
	}
	focus_distance = atof(argv[1]);
	return 0;
}

static char* console_aperture(console_ref ref, int argc, char **argv) {
	if (argc > 2)
		return strdup("without parameter: return lens aperture size, otherwise set.");
	if (argc == 1) {
		ostringstream oss; oss << "aperture = " << aperture;
		return strdup(oss.str().c_str());
	}
	aperture = atof(argv[1]);
	return 0;
}


string select_algo = "";

void actual_main() 
{
	dump_gl_info();
	for (int i = 0; i < time_measurement_samples; ++i)
		times[i] = 0.0f;

	register_display_function(display);
	register_idle_function(idle);

#ifdef WITH_GUILE
	register_cgls_scheme_functions();
	register_scheme_functions();
#endif

	initialize_interaction();
	push_interaction_mode(make_default_cgls_interaction_mode());
	push_interaction_mode(make_viewer_mode());

    gbuffer = make_stock_deferred_buffer("gbuffer", cmdline.res.x, cmdline.res.y, GL_RGBA8, GL_RGBA8, GL_RGBA16F, GL_RGBA32F, GL_DEPTH_COMPONENT24);

	for (list<string>::iterator it = cmdline.image_paths.begin(); it != cmdline.image_paths.end(); ++it)
		append_image_path(it->c_str());
	
	for (auto path : cmdline.image_paths)
		rta::append_image_path(path);
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images");
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images/sponza");


	scm_c_eval_string("(define gui #t)");

	if (cmdline.configs.size() != 0) {
		load_configfile("lights.scm");
		load_configfile("scenes.scm");
		scm_c_eval_string("(define gui #t)");
			for (int c = 0; c < cmdline.configs.size(); ++c)
				for (int p = 0; p < cmdline.include_paths.size(); ++p) {
					char *config = 0;
					int n = asprintf(&config, "%s/%s", cmdline.include_paths[p].c_str(), cmdline.configs[c].c_str());
					if (file_exists(config)) {
						load_configfile(config);
						free(config);
						break;
					}
					free(config);
				}
		scene_ref scene = { 0 };
		the_scene = scene;
		load_configfile("local.scm");
	}
	else
		cerr << "no config file given" << endl;
	
	struct drawelement_array picking_des = make_drawelement_array();
	push_drawelement_list_to_array(scene_drawelements(the_scene), &picking_des);
	picking = make_picking_buffer("pick", &picking_des, cmdline.res.x, cmdline.res.y);
	push_interaction_mode(make_blender_style_interaction_mode(the_scene, picking));
	
	vec3f up = vec3f(0, 1, 0);
	light_ref hemi = make_hemispherical_light("hemi", gbuffer, &up);
	change_light_color3f(hemi, .5, .5, .5);
	add_light_to_scene(the_scene, hemi);

	vec3f pos = vec3f(300, 200, 0);
	vec3f dir = vec3f(1, 0, 0);
	light_ref spot = make_spotlight("spot", gbuffer, &pos, &dir, &up, 30);
	add_light_to_scene(the_scene, spot);
	push_drawelement_to_array(light_representation(spot), &picking_des);

	pos = vec3f(700, 670, 0);
	dir = vec3f(0, -1, 0);
	up = vec3f(0, 0, 1);
// 	light_ref rect = make_rectangular_light("rect", gbuffer, &pos, &dir, &up, 50, 20);
// 	light_ref rect = make_rectangular_light("rect", gbuffer, &pos, &dir, &up, 100, 100);
// 	change_light_color3f(rect, 50, 50, 50);
// 	add_light_to_scene(the_scene, rect);
// 	push_drawelement_to_array(light_representation(rect), &picking_des);

	scene_set_lighting(the_scene, apply_deferred_lights);

	finalize_single_material_passes_for_array(&picking_des);

	viconsole = make_vi_console("vi-console", cmdline.res.x, cmdline.res.y);
	pop_interaction_mode();
	add_vi_console_command(viconsole, "screenshot", console_screenshot);
	add_vi_console_command(viconsole, "ss", console_screenshot);
	add_vi_console_command(viconsole, "bookmark", console_bookmark);
	add_vi_console_command(viconsole, "bm", console_bookmark);
	add_vi_console_command(viconsole, "b", console_select_bookmark);
	add_vi_console_command(viconsole, "a", console_algo);
	add_vi_console_command(viconsole, "decl", console_decl);
	add_vi_console_command(viconsole, "set", console_set);
	add_vi_console_command(viconsole, "show", console_show);
	add_vi_console_command(viconsole, "exp", console_exposure);
	add_vi_console_command(viconsole, "exposure", console_exposure);
	add_vi_console_command(viconsole, "aperture", console_aperture);
	add_vi_console_command(viconsole, "ap", console_aperture);
	add_vi_console_command(viconsole, "fd", console_focus_distance);
	add_vi_console_command(viconsole, "fod", console_focus_distance);
	push_interaction_mode(console_interaction_mode(viconsole));

	if (!cmdline.filename) {
		cout << "FILENAME NOT SET YET" << endl;
		exit(1);
	}
	char *base = basename((char*)cmdline.filename);
	string expr = string("(select-bookmark \"") + base + "\" \"start\")";
	SCM r = scm_c_eval_string(expr.c_str());

	setup_rta("bbvh-cuda");

	quad = make_quad("saq", 0);
	quad_shader = find_shader("show tex on saq");
	
	make_shader_error_display(cmdline.res.x, cmdline.res.y);
	reload_shaders();
	activate_automatic_shader_reload();

	new local::gpu_arealight_sampler(cmdline.res.x, cmdline.res.y, the_scene);
	new local::hybrid_arealight_sampler(cmdline.res.x, cmdline.res.y, the_scene);
	new local::gpu_cgls_lights(cmdline.res.x, cmdline.res.y, the_scene);
// 	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, 45.f, .5f, 5.f);
	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, focus_distance, aperture, 5.f);
// 	new gpu_pt(cmdline.res.x, cmdline.res.y, the_scene);

// 	gi_algorithm::select("gpu_cgls_lights");
// 	gi_algorithm::select("gpu_area_lights");
	if (select_algo == "")
		gi_algorithm::select("hybrid_area_lights");
	else
		gi_algorithm::select(select_algo);
// 	gi_algorithm::select("gpu_cgls_lights_dof");
// 	gi_algorithm::select("gpu_pt");

	init_rayvis(8, cmdline.res.x/200, cmdline.res.y/200);

	enter_glut_main_loop();
}

int main(int argc, char **argv)
{	
	parse_cmdline(argc, argv);
	
// 	int guile_mode = guile_cfg_only;
	int guile_mode = with_guile;
#ifndef WITH_GUILE
	guile_mode = without_guile;
#endif
	startup_cgl("name", 4, 2, argc, argv, (int)cmdline.res.x, (int)cmdline.res.y, actual_main, guile_mode, false, 0);

	return 0;
}

#endif

SCM_DEFINE(s_add_model, "add-model%", 6, 0, 0, (SCM filename, SCM type, SCM is_base, SCM trafo, SCM subd_disp, SCM subd_proxy), 
		   "internal function to load a model.") {
	char *file = scm_to_locale_string(filename);
	char *d_file = scm_to_locale_string(subd_disp);
	char *p_file = scm_to_locale_string(subd_proxy);
	int typecode = scm_to_int(type);
	bool base = scm_is_true(is_base);
	if (base)
		cmdline.filename = file;
	char *trf = scm_to_locale_string(trafo);
	if (typecode == 0) {
		// add_objfile_to_flat_tri_list(file, *ftl, trf);
		ostringstream oss;
		if (base) oss << "(load-base-obj ";
		else      oss << "(load-obj ";
		oss << "\""  << file << "\")";
		scm_c_eval_string(oss.str().c_str());
		return SCM_BOOL_T;
	}
	if (typecode == 1) {
		add_subd_model(file, d_file, p_file);
#if HAVE_LIBOSDINTERFACE != 1
		cerr << "Error: Support for SubD surfaces was not compiled in!" << endl;
#endif
		return SCM_BOOL_T;
	}
	cerr << "Error. Unknown model code (" << typecode << ")" << endl;
	free(trf);
	free(d_file);
	free(p_file);
	free(file);
	return SCM_BOOL_F;
}

SCM_DEFINE(s_subd_tess, "subd-tess", 2, 0, 0, (SCM n, SCM q), "set subd tesselation parameters") {
	subd_tess_normal = scm_to_int(n);
	subd_tess_quant = scm_to_int(q);
	return SCM_BOOL_T;
}

SCM_DEFINE(s_dof_config, "dof-config", 3, 0, 0, (SCM a, SCM d, SCM e), "dof parameters") {
	aperture = scm_to_double(a);
	focus_distance = scm_to_double(d);
	eye_to_lens = scm_to_double(e);
	return SCM_BOOL_T;
}
	
SCM_DEFINE(s_light_samples, "light-samples", 1, 0, 0, (SCM samples), "change number of the current algorithm's light samples (whatever that might mean, might be area light samples for direct illum)") {
	int s = scm_to_int(samples);
	s = max(1, s);
	if (gi_algorithm::selected)
		gi_algorithm::selected->light_samples(s);
	else
		init_light_samples = s;
	return SCM_BOOL_T;
}

SCM_DEFINE(s_path_samples, "path-samples", 1, 0, 0, (SCM samples), "change number of the current algorithm's path samples (whatever that might mean, might be multi sampling for pt)") {
	int s = scm_to_int(samples);
	s = max(1, s);
	if (gi_algorithm::selected)
		gi_algorithm::selected->path_samples(s);
	else
		init_path_samples = s;
	return SCM_BOOL_T;
}

SCM_DEFINE(s_path_len, "path-length", 1, 0, 0, (SCM samples), "change number of the current algorithm's light samples (whatever that might mean)") {
	int s = scm_to_int(samples);
	s = max(1, s);
	if (gi_algorithm::selected)
		gi_algorithm::selected->path_length(s);
	else
		init_path_length = s;
	return SCM_BOOL_T;
}

SCM_DEFINE(s_use_algo, "integrator", 1, 0, 0, (SCM name), "which algorithm to use") {
	char *n = scm_to_locale_string(name);
	select_algo = n;
	free(n);
	return SCM_BOOL_T;
}

SCM_DEFINE(s_disp_scale, "displacement-scale", 1, 0, 0, (SCM s), "set displacement scale factor (default: 1)") {
	subd_disp_scale = scm_to_double(s);
	return SCM_BOOL_T;
}

SCM_DEFINE(s_use_faces, "use-only-those-faces", 1, 0, 0, (SCM only), "") {
	char *n = scm_to_locale_string(only);
	subd_face_include = n;
	free(n);
	return SCM_BOOL_T;
}




static void register_scheme_functions_for_main() {
	#include "main.x"
}

