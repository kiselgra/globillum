#include <libcgls/cgls.h>
#include <libcgls/picking.h>
#include <libcgls/interaction.h>
#include <libcgls/console.h>
#include <libcgl/wall-time.h>

#include "cmdline.h"
#include "noscm.h"

#include "gi_algorithm.h"

#include <GL/freeglut.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

scene_ref the_scene;
#define samples 128
float times[samples];
int valid_pos = 0, curr_pos = 0;

framebuffer_ref gbuffer;
picking_buffer_ref picking;

console_ref viconsole = { -1 };

void display() {
	glDisable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEPTH_TEST);
	
    glFinish();
	wall_time_t start = wall_time_in_ms();

    bind_framebuffer(gbuffer);

	glClearColor(0,0,0.25,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	render_scene(the_scene);
    unbind_framebuffer(gbuffer);

	render_scene_deferred(the_scene, gbuffer);

	glFinish();
	wall_time_t end = wall_time_in_ms();

	times[curr_pos] = end-start;
	curr_pos = (curr_pos+1) % samples;
	valid_pos = (valid_pos == samples ? samples : valid_pos+1);

    check_for_gl_errors("end of display");

	render_console(viconsole);
	swap_buffers();
}

void idle() {
	glutPostRedisplay(); 
}

void show_fps(interaction_mode *m, int x, int y) {
	double sum = 0;
	for (int i = 0; i < valid_pos; ++i)
		sum += times[i];
	float avg = sum / (double)valid_pos;
	printf("average render time: %.3f ms, %.1f fps \t(sum %f, n %d)\n", avg, 1000.0f/avg, (float)sum, valid_pos);
}

void advance_anim(interaction_mode *m, int x, int y) {
	skeletal_animation_ref ar = { 0 };
	static float time = 0;
	time += 0.01;

	evaluate_skeletal_animation_at(ar, time);
}

interaction_mode* make_viewer_mode() {
	interaction_mode *m = make_interaction_mode("viewer");
	add_function_key_to_mode(m, 'p', cgls_interaction_no_button, show_fps);
	add_function_key_to_mode(m, ' ', cgls_interaction_no_button, advance_anim);
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
}
static void register_scheme_functions() {
	register_scheme_functions_for_cmdline();
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
	for (string name : names)
		if (name.find(argv[1]) != string::npos) {
			found = name;
			break;
		}
	if (found != "") {
		gi_algorithm::select(found);
		return strdup(found.c_str());
	}
	else return strdup("not found");
}



void actual_main() 
{
	dump_gl_info();
	for (int i = 0; i < samples; ++i)
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

	if (cmdline.config != "") {
#ifdef WITH_GUILE
		char *config = 0;
		int n = asprintf(&config, "%s/%s", cmdline.include_path, cmdline.config);
		load_configfile(config);
		free(config);
		scene_ref scene = { 0 };
		the_scene = scene;
#else
		scene::scene::select(cmdline.config);
		scene::scene::selected->load();
#endif
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
	push_interaction_mode(console_interaction_mode(viconsole));


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

