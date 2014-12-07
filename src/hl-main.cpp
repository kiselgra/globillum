#define PNG_SKIP_SETJMP_CHECK

#include <libobjloader/default.h>
#include <librta/material.h>
#include <libhyb/rta-cgls-connection.h>

#include "noscm.h"

#include "gi_algorithm.h"
#include "gpu_cgls_lights.h"
#include "gpu-pt.h"

#include "material.h"
#include "vars.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <argp.h>
#include <libgen.h>

using namespace std;

//// cmdline stuff

struct Cmdline {
	bool verbose;
	const char *filename;
	vec3f hemi_dir;
	bool hemi;
    const char *config;
    const char *include_path;
    vec2f res;
	bool scenefile, objfile;
	float merge_factor;
	std::list<std::string> image_paths;
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
	{ "merge-factor", MERGE, "x", 0, "Drawelement collapse threshold."},
	{ 0 }
};	

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

	switch (key)
	{
	case 'v':	cmdline.verbose = true; 	break;
    case 'c':   cmdline.config = strdup(arg); break;
    case 'I':   cmdline.include_path = strdup(arg); break;
    case 'i':   cmdline.image_paths.push_back(sarg); break;
    case 'r':   cmdline.res = read_vec2f(sarg); break;
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
    cmdline.config = strdup("default.c.scm");
    cmdline.include_path = DATADIR;
    cmdline.res.x = 1366; 
    cmdline.res.y = 768;
	cmdline.scenefile = cmdline.objfile = false;
	cmdline.merge_factor = 10;
	int ret = argp_parse(&parser, argc, argv, /*ARGP_NO_EXIT*/0, 0, 0);

	if (cmdline.filename == 0) {
		fprintf(stderr, "ERROR: no model or scene file specified. exiting...\n");
		exit(EXIT_FAILURE);
	}

	int dot = string(cmdline.filename).find_last_of(".");
	if (string(cmdline.filename).substr(dot) == ".obj")
		cmdline.objfile = true;
	else if (string(cmdline.filename).substr(dot) == ".bobj")
		cmdline.objfile = true;
	else
		cmdline.scenefile = true;
	return ret;
}
	



//// misc globals

std::map<std::string, var> vars;

float exposure = 10;

// used for dof implementations
float aperture = .5;
float focus_distance = 970.0f;

scene_ref the_scene;


//// rta setup

static rta::cgls::connection *rta_connection = 0;
rta::basic_flat_triangle_list<rta::simple_triangle> *ftl = 0;
rta::cgls::connection::cuda_triangle_data *ctd = 0;
rta::cuda::material_t *gpu_materials = 0;

rta::basic_flat_triangle_list<rta::simple_triangle> load_objfile_to_flat_tri_list(const std::string &filename) {
	obj_default::ObjFileLoader loader(filename, "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1");

	int triangles = 0;
	for (auto &group : loader.groups)
		triangles += group.load_idxs_v.size();

	rta::basic_flat_triangle_list<rta::simple_triangle> ftl(triangles);

	for (auto path : cmdline.image_paths)
		rta::append_image_path(path);
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images");
	rta::append_image_path(string(getenv("HOME")) + "/render-data/images/sponza");
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
				rta::material_t *mat = new rta::material_t(group.mat->name, d, group.mat->tex_d);
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
}


void setup_rta(std::string plugin) {
	bool use_cuda = true;
	vector<string> args;
	if (plugin == "default/choice")
		if (use_cuda) 
			plugin = "bbvh-cuda";
		else
			plugin = "bbvh";
			
	if (plugin == "bbvh-cuda") {
		args.push_back("-A");
		args.push_back("-b");
		args.push_back("bsah");
		args.push_back("-t");
		args.push_back("cis");
		args.push_back("-l");
		args.push_back("2f4");
	}

	/*
	rta_connection = new rta::cgls::connection(plugin, args);
	ctd = rta::cgls::connection::convert_scene_to_cuda_triangle_data(the_scene);
	static rta::basic_flat_triangle_list<rta::simple_triangle> the_ftl = ctd->cpu_ftl();
	*/
	int rays_w = cmdline.res.x, rays_h = cmdline.res.y;
	static rta::basic_flat_triangle_list<rta::simple_triangle> the_ftl = load_objfile_to_flat_tri_list(cmdline.filename);
	ftl = &the_ftl;
	rta::rt_set *set = new rta::rt_set(rta::plugin_create_rt_set(*ftl, rays_w, rays_h));
	/*
	gpu_materials = rta::cuda::convert_and_upload_materials();

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

int main(int argc, char **argv)
{	
	parse_cmdline(argc, argv);

#ifdef WITH_GUILE
	register_cgls_scheme_functions();
#endif

	for (list<string>::iterator it = cmdline.image_paths.begin(); it != cmdline.image_paths.end(); ++it)
		append_image_path(it->c_str());

	if (cmdline.config != "") {
#ifdef WITH_GUILE
		char *config = 0;
		int n = asprintf(&config, "%s/%s", cmdline.include_path, cmdline.config);
		load_configfile(config);
		free(config);
		load_configfile("local.scm");
#endif
	}
	else
		cerr << "no config file given" << endl;
	
	char *base = basename((char*)cmdline.filename);
	string expr = string("(select-bookmark \"") + base + "\" \"start\")";
	SCM r = scm_c_eval_string(expr.c_str());

	setup_rta("bbvh-cuda");

	new local::gpu_cgls_lights_arealight_sampler(cmdline.res.x, cmdline.res.y, the_scene);
	new local::gpu_cgls_lights(cmdline.res.x, cmdline.res.y, the_scene);
// 	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, 45.f, .5f, 5.f);
	new local::gpu_cgls_lights_dof(cmdline.res.x, cmdline.res.y, the_scene, focus_distance, aperture, 5.f);
// 	new gpu_pt(cmdline.res.x, cmdline.res.y, the_scene);

// 	gi_algorithm::select("gpu_cgls_lights");
	gi_algorithm::select("gpu_cgls_lights_dof");
// 	gi_algorithm::select("gpu_pt");


	// START COMPUTATION
	
	return 0;
}

