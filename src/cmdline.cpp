#include "cmdline.h"


#include <argp.h>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdlib.h>



using namespace std;

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
	else
		cmdline.scenefile = true;
	return ret;
}
	
Cmdline cmdline;


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

	void register_scheme_functions_for_cmdline() {
		#include "cmdline.x"
	}
}

#endif
