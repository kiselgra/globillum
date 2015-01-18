#include "subd.h"

#include <libhyb/rta-cgls-connection.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <libguile.h>

using namespace std;

#if HAVE_LIBOSDINTERFACE == 1
vector<OSDI::Model*> subd_models;
#endif

// for ptex
static vector<string> subd_files;
static vector<string> disp_files;
static vector<string> proxy_files;
static vector<string> occlusion_files;
static vector<string> pose_files;
static vector<string> spec_files;

// for obj-mod
static vector<string> objmodel_files;
// static vector<string> subdobjproxy_files;

int subd_tess_normal = 1;
int subd_tess_quant = 1;
float subd_disp_scale = 1.0f;
std::string subd_face_include;

void add_subd_model(const std::string &filename, const std::string &displacement, const std::string &proxy, const std::string &pose, const std::string &spec, const std::string &occ) {
	cout << "add sub model " << filename << endl;
	subd_files.push_back(filename);
	disp_files.push_back(displacement);
	pose_files.push_back(pose);
	spec_files.push_back(spec);
	occlusion_files.push_back(occ);
	if (proxy != "") proxy_files.push_back(proxy);
}
	
void add_subd_obj_model(const std::string &filename, const std::string &proxy) {
	cout << "add sub/obj model " << filename << endl;
	objmodel_files.push_back(filename);
	if (proxy != "") proxy_files.push_back(proxy);
}
	
void load_subd_proxies() {
	for (string m : proxy_files) {
		ostringstream oss;
		oss << "(load-obj \"" << m << "\")";
		scm_c_eval_string(oss.str().c_str());
	}
}

// --model-obj
// --occlu=filename       oclcusion file
// --pose=filename        set model pose with file


#if HAVE_LIBOSDINTERFACE == 1
rta::rt_set* generate_compressed_bvhs_and_tracer(int w, int h) {
	if (subd_files.size() == 0) return 0;
	vector<string> args;
	ostringstream n; n << subd_tess_normal;
	ostringstream q; q << subd_tess_quant;
	ostringstream f; f << subd_disp_scale;

	args.push_back("-n");
	args.push_back(n.str());
	args.push_back("-q");
	args.push_back(q.str());
	args.push_back("--node");
// 	args.push_back("test");
	args.push_back("uni662");
	if (subd_face_include != "") {
		args.push_back("-f");
		args.push_back(subd_face_include);
	}
	args.push_back("--scale");
	args.push_back(f.str());
	for (int i = 0; i < subd_files.size(); ++i) {
		args.push_back("--model");
		args.push_back(subd_files[i]);
		args.push_back("--displ");
		args.push_back(disp_files[i]);
		if (occlusion_files[i] != "") {
			args.push_back("--occlu");
			args.push_back(occlusion_files[i]);
		}
		if (spec_files[i] != "") {
			args.push_back("--spec");
			args.push_back(spec_files[i]);
		}
		if (pose_files[i] != "") {
			args.push_back("--pose");
			args.push_back(pose_files[i]);
		}
	}
	for (int i = 0; i < objmodel_files.size(); ++i) {
		args.push_back("--model-obj");
		args.push_back(objmodel_files[i]);
	}
	// load subd plugin
	rta::cgls::connection rta_connection("subdiv", args);
	rta::basic_flat_triangle_list<rta::simple_triangle> fake;
	rta::rt_set *set = new rta::rt_set(rta::plugin_create_rt_set(fake, w, h));

	subd_models = dynamic_cast<rta::model_holder*>(set->bouncer)->models;
	
	return set;
}
#endif


/* vim: set foldmethod=marker: */

