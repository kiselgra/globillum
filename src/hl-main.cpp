#define PNG_SKIP_SETJMP_CHECK
#ifndef SCM_MAGIC_SNARFER
#include <libobjloader/default.h>
#include <librta/material.h>
#include <libhyb/rta-cgls-connection.h>
#include <termios.h>
#include <unistd.h>

#include "noscm.h"
#include "config.h"

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"
#include "gpu-pt.h"
#include "hybrid-pt.h"
#include "cpu-pt.h"

#include "subd.h"

#include "material.h"
#include "vars.h"
#include "util.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <argp.h>
#include <libgen.h>

using namespace std;


#define CAMERA_PATHS 0

//// cmdline stuff

struct Cmdline {
	bool verbose;
	const char *filename;
	vec3f hemi_dir;
	bool hemi;
	std::vector<std::string> configs;
	std::vector<std::string> include_paths;
    vec2f res;
	bool scenefile, objfile;
	float merge_factor;
	std::list<std::string> image_paths;
	bool lazy;
} cmdline;
const char *argp_program_version = VERSION;

static char doc[]       = PACKAGE ": description";
static char args_doc[]  = "argumentdescription";

// long option without corresponding short option have to define a symbolic constant >= 300
enum { FIRST = 300, MERGE, OPTS };

static struct argp_option options[] = 
{
	// --[opt]		short/const		arg-descr		?		option-descr
	{ "verbose", 'v', 0,         0, "Be verbose." },
    { "config", 'c', "configfile", 0, "Scheme file to supply instructions."},
    { "include-path", 'I', "path", 0, "Path to search for the config file. Default: " DATADIR "."},
    { "image-path", 'i', "path", 0, "Path to search for images. May be specified multiple times."},
	{ "res", 'r', "w,h", 0, "Window resolution."},
	{ "prefix", 'P', "path", 0, "Path prefix to store output images. Default: /tmp/"},
	{ "merge-factor", MERGE, "x", 0, "Drawelement collapse threshold."},
	{ "lazy", 'l', 0, 0, "Don't start computation right ahead."},
	{ "output-format", 'F', "p, e", 0, "Save png files or exr files. Can be specified multiple times."},
	{ 0 }
};	
std::vector<std::string> subdFilenames;
string& replace_nl(string &s)
{
	for (int i = 0; i < s.length(); ++i)
		if (s[i] == '\n' || s[i] == '\r')
			s[i] = ' ';
	return s;
}

vec3f read_vec3f(const std::string &s) {
	istringstream iss(s);
	vec3f v;
	char sep;
	iss >> v.x >> sep >> v.y >> sep >> v.z;
	return v;
}

vec2f read_vec2f(const std::string &s) {
	istringstream iss(s);
	vec2f v;
	char sep;
	iss >> v.x >> sep >> v.y;
	return v;
}

static error_t parse_options(int key, char *arg, argp_state *state)
{
	// call argp_usage to stop program execution if something is wrong
	
	string sarg;
	if (arg)
		sarg = arg;
	sarg = replace_nl(sarg);
	static bool output_format_init = true;

	switch (key)
	{
	case 'v':	cmdline.verbose = true; 	break;
    case 'c':   cmdline.configs.push_back(sarg); break;
    case 'I':   cmdline.include_paths.push_back(sarg); break;
    case 'l':   cmdline.lazy = true; break;
    case 'i':   cmdline.image_paths.push_back(sarg); break;
    case 'r':   cmdline.res = read_vec2f(sarg); break;
	case 'P':   gi::image_store_path = sarg; 
				if (gi::image_store_path[gi::image_store_path.length()-1] != '/')
					gi::image_store_path += "/";
				break;
	case 'F':   if (output_format_init) { output_format_init = false; gi::image_output_format = 0; }
				if (sarg == "p")
					gi::image_output_format |= gi::output_format::png;
				else if (sarg == "e")
					gi::image_output_format |= gi::output_format::exr;
				else
					cerr << "unknown image output format '" << sarg << "'" << endl;
	case MERGE: cmdline.merge_factor = atof(arg); break;
	
	case ARGP_KEY_ARG:		// process arguments. 
							// state->arg_num gives number of current arg
				if (cmdline.filename)
					fprintf(stderr, "ERROR: you can display only one model at a time.\n");
				cmdline.filename = strdup(arg);
		break;

	default:
		return ARGP_ERR_UNKNOWN;
	}

	return 0;
}

static struct argp parser = { options, parse_options, args_doc, doc };

int parse_cmdline(int argc, char **argv)
{
	cmdline.filename = 0;
    cmdline.res.x = 1366; 
    cmdline.res.y = 768;
	cmdline.scenefile = cmdline.objfile = false;
	cmdline.merge_factor = 10;
	cmdline.lazy = false;
	int ret = argp_parse(&parser, argc, argv, /*ARGP_NO_EXIT*/0, 0, 0);
    
	if (cmdline.configs.size() == 0)
		cmdline.configs.push_back("nogui.scm");
	cmdline.include_paths.push_back(".");

	if (cmdline.filename == 0) {
// 		fprintf(stderr, "ERROR: no model or scene file specified. exiting...\n");
// 		exit(EXIT_FAILURE);
	}
	else {
		int dot = string(cmdline.filename).find_last_of(".");
		if (string(cmdline.filename).substr(dot) == ".obj")
			cmdline.objfile = true;
		else if (string(cmdline.filename).substr(dot) == ".bobj")
			cmdline.objfile = true;
		else
			cmdline.scenefile = true;
	}
	return ret;
}
	



//// misc globals

std::map<std::string, var> vars;

float exposure = 10;

// used for dof implementations
float aperture = .0f;
float focus_distance = 970.0f;
float eye_to_lens = 5.f;

scene_ref the_scene = { -1 };

extern int subd_tess_normal;
extern int subd_tess_quant;
extern float subd_disp_scale;
extern std::string subd_face_include;

//// rta setup

static rta::cgls::connection *rta_connection = 0;
rta::basic_flat_triangle_list<rta::simple_triangle> *ftl = 0;
rta::cgls::connection::cuda_triangle_data *ctd = 0;
int material_count = 0;

int curr_frame = 0;

int idx_subd_material = 0;
rta::cuda::material_t *gpu_materials = 0;
rta::cuda::material_t *cpu_materials = 0;

//! this is a direct copy of rta code.
/*
rta::basic_flat_triangle_list<rta::simple_triangle> load_objfile_to_flat_tri_list(const std::string &filename) {
	obj_default::ObjFileLoader loader(filename, "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1");

	int triangles = 0;
	for (auto &group : loader.groups)
		triangles += group.load_idxs_v.size();

	rta::basic_flat_triangle_list<rta::simple_triangle> ftl(triangles);

	rta::prepend_image_path(dirname((char*)filename.c_str()));

	int run = 0;
	for (auto &group : loader.groups) {
		// building those is expensive!
		auto vertex   = [=](int id, int i) { auto v = loader.load_verts[((int*)&group.load_idxs_v[i])[id]]; vec3f r = {v.x,v.y,v.z}; return r; };
		auto normal   = [=](int id, int i) { auto v = loader.load_norms[((int*)&group.load_idxs_n[i])[id]]; vec3f r = {v.x,v.y,v.z}; return r; };
		auto texcoord = [=](int id, int i) { auto v = loader.load_texs[((int*)&group.load_idxs_t[i])[id]]; vec2f r = {v.x,v.y}; return r; };
		int t = group.load_idxs_v.size();
		int mid = -1;
		if (group.mat) {
			mid = rta::material(group.mat->name);
			if (mid == -1) {
				vec3f d = { group.mat->dif_r, group.mat->dif_g, group.mat->dif_b };
				vec3f s = { group.mat->spe_r, group.mat->spe_g, group.mat->spe_b };
				rta::material_t *mat = new rta::material_t(group.mat->name, d, group.mat->tex_d);
				mat->specular_color = s;
				if (group.mat->tex_s != "")
					mat->add_specular_texture(group.mat->tex_s);
				mat->alpha = group.mat->alpha;
				if (group.mat->tex_alpha != "")
					mat->add_alpha_texture(group.mat->tex_alpha);
				mid = rta::register_material(mat);
			}
		}
		for (int i = 0; i < t; ++i)	{
			ftl.triangle[run + i].a = vertex(0, i);
			ftl.triangle[run + i].b = vertex(1, i);
			ftl.triangle[run + i].c = vertex(2, i);
			ftl.triangle[run + i].na = normal(0, i);
			ftl.triangle[run + i].nb = normal(1, i);
			ftl.triangle[run + i].nc = normal(2, i);
			if (mid >= 0 && rta::material(mid)->diffuse_texture) {
				ftl.triangle[run + i].ta = texcoord(0, i);
				ftl.triangle[run + i].tb = texcoord(1, i);
				ftl.triangle[run + i].tc = texcoord(2, i);
			}
			else {
				ftl.triangle[run + i].ta = {0,0};
				ftl.triangle[run + i].tb = {0,0};
				ftl.triangle[run + i].tc = {0,0};
			}
			ftl.triangle[run + i].material_index = mid;
		}
		run += t;
	}

	rta::pop_image_path_front();

	return ftl;
}*/

void add_objfile_to_flat_tri_list(const std::string &filename, rta::basic_flat_triangle_list<rta::simple_triangle> &ftl, const char *trafo) {
	obj_default::ObjFileLoader loader(filename, trafo); //"1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1";

	int triangles = 0;
	for (auto &group : loader.groups)
		triangles += group.load_idxs_v.size();

// 	rta::basic_flat_triangle_list<rta::simple_triangle> ftl(triangles);
	rta::simple_triangle *old = ftl.triangle;
	ftl.triangle = new rta::simple_triangle[ftl.triangles+triangles];
	memcpy(ftl.triangle, old, sizeof(rta::simple_triangle)*ftl.triangles);
	delete [] old;
	int offset = ftl.triangles;
	ftl.triangles += triangles;
	cout << "ftl with " << ftl.triangles << " tris now" << endl;

	rta::prepend_image_path(dirname((char*)filename.c_str()));

	if (offset == 0) {
		// for the first obj loaded, create a fallback material for all objects that have no material
		vec3f d = {1,0,0};
		vec3f s = {1,0,0};
		rta::material_t *mat = new rta::material_t("gi/fallback", d);
		mat->specular_color = s;
		rta::register_material(mat);
	}

	int run = 0;
	for (auto &group : loader.groups) {
		// building those is expensive!
		auto vertex   = [=](int id, int i) { auto v = loader.load_verts[((int*)&group.load_idxs_v[i])[id]]; vec3f r = {v.x,v.y,v.z}; return r; };
		auto normal   = [=](int id, int i) { auto v = loader.load_norms[((int*)&group.load_idxs_n[i])[id]]; vec3f r = {v.x,v.y,v.z}; return r; };
		auto texcoord = [=](int id, int i) { auto v = loader.load_texs[((int*)&group.load_idxs_t[i])[id]]; vec2f r = {v.x,v.y}; return r; };
		int t = group.load_idxs_v.size();
		int mid = -1;
		if (group.mat) {
			mid = rta::material(group.mat->name);
			if (mid == -1) {
				vec3f d = { group.mat->dif_r, group.mat->dif_g, group.mat->dif_b };
				vec3f s = { group.mat->spe_r, group.mat->spe_g, group.mat->spe_b };
				rta::material_t *mat = new rta::material_t(group.mat->name, d, group.mat->tex_d);
				mat->specular_color = s;
				if (group.mat->tex_s != "")
					mat->add_specular_texture(group.mat->tex_s);
				mat->alpha = group.mat->alpha;
				if (group.mat->tex_alpha != "")
					mat->add_alpha_texture(group.mat->tex_alpha);
				mid = rta::register_material(mat);
			}
		}
		else
			mid = rta::material("gi/fallback");
		
		for (int i = 0; i < t; ++i)	{
			int pos = offset + run + i;
			ftl.triangle[pos].a = vertex(0, i);
			ftl.triangle[pos].b = vertex(1, i);
			ftl.triangle[pos].c = vertex(2, i);
			ftl.triangle[pos].na = normal(0, i);
			ftl.triangle[pos].nb = normal(1, i);
			ftl.triangle[pos].nc = normal(2, i);
			if (mid >= 0 && rta::material(mid)->diffuse_texture) {
				ftl.triangle[pos].ta = texcoord(0, i);
				ftl.triangle[pos].tb = texcoord(1, i);
				ftl.triangle[pos].tc = texcoord(2, i);
			}
			else {
				ftl.triangle[pos].ta = {0,0};
				ftl.triangle[pos].tb = {0,0};
				ftl.triangle[pos].tc = {0,0};
			}
			ftl.triangle[pos].material_index = mid;
		}
		run += t;
	}

	rta::pop_image_path_front();
}


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
		use_cuda = true;
		args.push_back("-A");
		args.push_back("-b");
		args.push_back("bsah");
// 		args.push_back("median");
		args.push_back("-t");
		args.push_back("cis");
		args.push_back("-l");
		args.push_back("2f4");
	}
	else if (plugin == "bbvh") {
		use_cuda = false;
	}

	rta_connection = new rta::cgls::connection(plugin, args);

	//go over all subd file paths and get their actual filename 
	// this name will be used to identify wether we have a corresponding material filename .pbrdf
	for(int i=0; i<subdFilenames.size(); i++){
		int idx = subdFilenames[i].rfind('/');
		if(idx != std::string::npos){
			subdFilenames[i]  = subdFilenames[i].substr(idx+1, subdFilenames[i].size() - idx - 5);
		}
	}
	/*
	ctd = rta::cgls::connection::convert_scene_to_cuda_triangle_data(the_scene);
	static rta::basic_flat_triangle_list<rta::simple_triangle> the_ftl = ctd->cpu_ftl();
	*/
// 	static rta::basic_flat_triangle_list<rta::simple_triangle> the_ftl = load_objfile_to_flat_tri_list(cmdline.filename);
// 	add_objfile_to_flat_tri_list(cmdline.filename, *ftl);
// 	ftl = &the_ftl;
	rta::rt_set *set = new rta::rt_set(rta::plugin_create_rt_set(*ftl, rays_w, rays_h));
	gpu_materials = rta::cuda::convert_and_upload_materials(material_count,subdFilenames);
	cpu_materials = rta::cuda::download_materials(gpu_materials, material_count);
	idx_subd_material = material_count - subdFilenames.size();
	/*

	if (!use_cuda) {
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
	}

	tex_params_t p = default_fbo_tex_params();
	gi_algorithm::result = make_empty_texture("gi-res", cmdline.res.x, cmdline.res.y, GL_TEXTURE_2D, GL_RGBA32F, GL_FLOAT, GL_RGBA, &p);
	*/
	
	if (!use_cuda) {
		if (!set->basic_ctor<rta::simple_aabb, rta::simple_triangle>()->expects_host_triangles()) {
			cerr << "does not want host tris!" << endl;
			exit(-1);
		}
	}
	else {
		if (!set->basic_ctor<rta::cuda::simple_aabb, rta::cuda::simple_triangle>()->expects_host_triangles()) {
			cerr << "does not want host tris!" << endl;
			exit(-1);
		}
	}
	
	gi_algorithm::original_rt_set = set;
}

//// guile/console hacks

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
		gi_algorithm::selected->debug(cmdline.verbose);
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

//// 

extern "C" {
	void register_scheme_functions_for_light_setup();
	void register_scheme_functions_for_cmdline();
}

static bool quit_loop = false;
static bool restart_compute = true;
static char *change_algo = 0;
static struct termios termios;
string select_algo = "";

void actual_main() {
	register_cgls_scheme_functions();
	register_scheme_functions_for_light_setup();
	register_scheme_functions_for_cmdline();

	ftl = new rta::basic_flat_triangle_list<rta::simple_triangle>;

// 	for (list<string>::iterator it = cmdline.image_paths.begin(); it != cmdline.image_paths.end(); ++it)
// 		append_image_path(it->c_str());
	
	for (auto path : cmdline.image_paths)
		rta::append_image_path(path);
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images");
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images/sponza");

	ostringstream oss; oss << "(define x-res " << cmdline.res.x << ") (define y-res " << cmdline.res.y << ")";
	scm_c_eval_string(oss.str().c_str());
	load_configfile("lights.scm");
	load_configfile("scenes.scm");
	scm_c_eval_string("(define gui #f)");
	for (int c = 0; c < cmdline.configs.size(); ++c)
		for (int p = 0; p < cmdline.include_paths.size(); ++p) {
			char *config = 0;
			int n = asprintf(&config, "%s/%s", cmdline.include_paths[p].c_str(), cmdline.configs[c].c_str());
			if (file_exists(config)) {
				cout << config << endl;
				load_configfile(config);
				free(config);
				break;
			}
			free(config);
		}
	load_configfile("local.scm");

	char *base = basename((char*)cmdline.filename);
	string expr = string("(select-bookmark \"") + base + "\" \"start\")";
	SCM r = scm_c_eval_string(expr.c_str());
	expr = string("(define scene \"") + base + "\")";
	r = scm_c_eval_string(expr.c_str());
	scm_c_eval_string("(define (lbm) (list-bookmarks scene))");
	scm_c_eval_string("(define (b n) (if (not (select-bookmark scene n)) (format #t \"Error: No such bookmark.~%\")))");

	if (select_algo == "cpu_pt")
		setup_rta("bbvh");
	else
		setup_rta("bbvh-cuda");

	new local::gpu_arealight_sampler(cmdline.res.x, cmdline.res.y, the_scene);
	new local::hybrid_arealight_sampler(cmdline.res.x, cmdline.res.y, the_scene);
	new local::gpu_cgls_lights(cmdline.res.x, cmdline.res.y, the_scene);
// 	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, 45.f, .5f, 5.f);
	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, focus_distance, aperture, 5.f);
// 	new gpu_pt(cmdline.res.x, cmdline.res.y, the_scene);
	new hybrid_pt(cmdline.res.x, cmdline.res.y, the_scene);
	new cpu_pt(cmdline.res.x, cmdline.res.y, the_scene);

// 	gi_algorithm::select("gpu_cgls_lights");
// 	gi_algorithm::select("gpu_area_lights");
// 	gi_algorithm::select("gpu_cgls_lights_dof");
// 	gi_algorithm::select("gpu_pt");
// 	gi_algorithm::select("hybrid_area_lights");
	if (select_algo == "")
		gi_algorithm::select("hybrid_pt");
	else
		gi_algorithm::select(select_algo);

	scm_c_eval_string("(set! gi-initialization-done #t)");

	// START COMPUTATION
	gi_algorithm *algo = gi_algorithm::selected;
	char *argv[5];
	console_ref console = {-1};
	restart_compute = !cmdline.lazy;
	curr_frame = 0;
#if CAMERA_PATHS
	int maxFrames = 360;
	vec3f midPoint;
	float phi = 0.f;// * M_PI/180.f;
	bool first_rotation = true;
	float phiAdd = 360.f/float(maxFrames) * M_PI/180.f;
#endif

	while (true) {
		if (restart_compute) {
			if (change_algo) {
				argv[0] = (char*)"a";
				argv[1] = change_algo;
				char *res = console_algo(console, 1, argv);
				if (res != 0) {
					cerr << "Error: " << res << endl;
					free(res);
				}
				free(change_algo);
				change_algo = 0;
			}
			algo->compute();
			restart_compute = false;
		}
		while (algo->progressive() && algo->in_progress()) {
			algo->update();
			if (quit_loop) break;
		}

#if CAMERA_PATHS
		camera_ref cam = current_camera();
		matrix4x4f *lookat_matrix = lookat_matrix_of_cam(cam);
		//compute new angle
		phi = phi + phiAdd;// M_PI/180.f;

		vec3f nextUp; //(0.f,1.f,0.f);
		nextUp.x = 0.f;
		nextUp.y = 1.0f;
		nextUp.z = 0.0f;
		//look at point is the same as before.
		vec3f nextDir;

		vec3f xAxisTest(1.f,0.f,0.f);
		vec3f yAxisTest(0.f,0.f,1.f);

		vec3f testPos(0.f,230.f,0.f);
		vec3f lookAtPos(0.f,150.f,0.f);

		float rad = 290;
		vec3f nextPos = testPos + xAxisTest * rad * sin(phi) + yAxisTest*cos(phi)*rad;
		nextDir = lookAtPos-nextPos;		
		normalize_vec3f(&nextDir);
		make_lookat_matrixf(lookat_matrix, &nextPos, &nextDir, &nextUp);//miatrix4x4d *out, const vec3d *pos, const vec3d *dir, const vec3d *up);
		
		curr_frame++;
		// restart rendering
		if(curr_frame < maxFrames)
			restart_compute = true;
		else
			quit_loop;
#endif
		// setup next frame.
		if (quit_loop) break;
		usleep(100000);
	}
	if (isatty(STDOUT_FILENO))
		tcsetattr(STDOUT_FILENO, TCSANOW, &termios);
}

extern "C" {
	void load_internal_configfiles(void);
}

//! this is a direct copy of cgl code.
static void hop(void *data, int argc, char **argv) {
#ifdef WITH_GUILE
	load_snarfed_definitions();
// 	load_internal_configfiles();
	if (argv[0]) load_configfile(argv[0]);
// 	start_console_thread();

	scm_c_eval_string("(define gi-initialization-done #f)");
	scm_c_eval_string("(define repl-thread (call-with-new-thread (lambda () "
	                                          "(use-modules (ice-9 readline)) "
											  "(activate-readline) "
											  "(while (not gi-initialization-done) (yield))"
											  "(format #t \"we're ready now.~%\")"
											  "(top-repl) "
											  "(q))))");

#endif

	((void(*)())data)();    // run the user supplied 'inner main'
}

int main(int argc, char **argv)
{	
	if (isatty(STDOUT_FILENO))
		tcgetattr(STDOUT_FILENO, &termios);

	parse_cmdline(argc, argv);

	char *p[1] = { 0 };
	scm_boot_guile(0, p, hop, (void*)actual_main);

	return 0;
}


#endif
#ifdef WITH_GUILE

#include <libguile.h>
#include <libcgl/scheme.h>

extern "C" {

	SCM_DEFINE(s_cmdline, "query-cmdline", 1, 0, 0, (SCM what), "") {
		if (!scm_is_symbol(what))
			scm_throw(scm_from_locale_symbol("cmdline-error"), scm_list_2(what, scm_from_locale_string("is not a symbol")));
		char *w = scm_to_locale_string(scm_symbol_to_string(what));
		string s = w;
		free(w);
		if (s == "model") {
			if (cmdline.objfile)
				return scm_from_locale_string(cmdline.filename);
			scm_throw(scm_from_locale_symbol("cmdline-error"), 
			          scm_list_2(what, 
			                    scm_from_locale_string("the program was invoked with a scene file, not a model file.")));
		}
		else if (s == "scene") {
			return scm_from_locale_string(cmdline.filename);
		}
		else if (s == "filetype") {
			return scm_string_to_symbol(scm_from_locale_string((cmdline.objfile ? string("obj") : string("scene")).c_str()));
		}
		else if (s == "merge-factor") {
			return scm_from_double(cmdline.merge_factor);
		}


		scm_throw(scm_from_locale_symbol("cmdline-error"), 
		          scm_list_2(what, 
		                    scm_from_locale_string("invalid option. use scene, model, filetype")));
	}

	SCM_DEFINE(s_quit, "quit", 0, 0, 0, (), "terminate interactive loop") {
		quit_loop = true;
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_recomp, "recompute", 0, 0, 0, (), "restart selected algorithm") {
		restart_compute = true;
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_select, "select", 1, 0, 0, (SCM namepart), "select different algorithm and start computation") {
		restart_compute = true;
		change_algo = scm_to_locale_string(namepart);
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

	SCM_DEFINE(s_subd_tess, "subd-tess", 2, 0, 0, (SCM n, SCM q), "set subd tesselation parameters") {
		subd_tess_normal = scm_to_int(n);
		subd_tess_quant = scm_to_int(q);
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_disp_scale, "displacement-scale", 1, 0, 0, (SCM s), "set displacement scale factor (default: 1)") {
		subd_disp_scale = scm_to_double(s);
		return SCM_BOOL_T;
	}
	SCM_DEFINE(s_dof_config, "dof-config", 3, 0, 0, (SCM a, SCM d, SCM e), "dof parameters") {
		aperture = scm_to_double(a);
		focus_distance = scm_to_double(d);
		eye_to_lens = scm_to_double(e);
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_use_algo, "integrator", 1, 0, 0, (SCM name), "which algorithm to use") {
		char *n = scm_to_locale_string(name);
		select_algo = n;
		free(n);
		return SCM_BOOL_T;
	}


	SCM_DEFINE(s_use_faces, "use-only-those-faces", 1, 0, 0, (SCM only), "") {
		char *n = scm_to_locale_string(only);
		subd_face_include = n;
		free(n);
		return SCM_BOOL_T;
	}

	SCM_DEFINE(s_add_model, "add-model%", 9, 0, 0, (SCM filename, SCM type, SCM is_base, SCM trafo, SCM subd_disp, SCM subd_proxy, SCM spec, SCM occl, SCM pose), 
			   "internal function to load a model.") {
		(void)subd_proxy;
		char *file = scm_to_locale_string(filename);
		char *d_file = scm_to_locale_string(subd_disp);
		char *p_file = scm_to_locale_string(subd_proxy);
		char *s_file = scm_to_locale_string(spec);
		char *o_file = scm_to_locale_string(occl);
		char *pose_file = scm_to_locale_string(pose);
		int typecode = scm_to_int(type);
		bool base = scm_is_true(is_base);
		if (base)
			cmdline.filename = file;
		char *trf = scm_to_locale_string(trafo);
		if (typecode == 0) {
			add_objfile_to_flat_tri_list(file, *ftl, trf);
			return SCM_BOOL_T;
		}
		if (typecode == 1) {
#if HAVE_LIBOSDINTERFACE == 1
			subdFilenames.push_back(file);
			add_subd_model(file, d_file, p_file, pose_file, s_file, o_file);
			return SCM_BOOL_T;
#else
			cerr << "Error: Support for SubD surfaces was not compiled in!" << endl;
#endif
		}
		cerr << "Error. Unknown model code (" << typecode << ")" << endl;
		free(trf);
		free(file);
		free(d_file);
		free(p_file);
		free(s_file);
		free(o_file);
		free(pose_file);
		return SCM_BOOL_F;
	}

	void register_scheme_functions_for_cmdline() {
		#include "hl-main.x"
		scm_c_eval_string("(define exit quit)");
		scm_c_eval_string("(define q quit)");
	}
}

#endif
