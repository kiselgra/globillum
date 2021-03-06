/* $Id$ */
#ifndef __CMDLINE_H__ 
#define __CMDLINE_H__ 

#include <libcgls/cgls.h>
#include <string>
#include <list>
#include <vector>

//! \brief Translated command line options
typedef struct
{
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
} Cmdline;

#ifdef __cplusplus
extern "C" {
#endif

	extern Cmdline cmdline;
	int parse_cmdline(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif

