#include "subd.h"

#include <libhyb/rta-cgls-connection.h>

#include <iostream>
#include <vector>

using namespace std;

static vector<string> subd_files;
static vector<string> disp_files;

void add_subd_model(const std::string &filename, const std::string &displacement) {
	cout << "add sub model " << filename << endl;
	subd_files.push_back(filename);
	disp_files.push_back(displacement);
}

rta::rt_set* generate_compressed_bvhs_and_tracer(int w, int h) {
	if (subd_files.size() == 0) return 0;
	vector<string> args;

	args.push_back("-n");
	args.push_back("1");
	args.push_back("-q");
	args.push_back("1");
	args.push_back("--node");
	args.push_back("test");
	for (int i = 0; i < subd_files.size(); ++i) {
		args.push_back("--model");
		args.push_back(subd_files[i]);
		args.push_back("--displ");
		args.push_back(disp_files[i]);
	}
	// load subd plugin
	rta::cgls::connection rta_connection("subdiv", args);
	rta::basic_flat_triangle_list<rta::simple_triangle> fake;
	rta::rt_set *set = new rta::rt_set(rta::plugin_create_rt_set(fake, w, h));

	return set;
}


/* vim: set foldmethod=marker: */

