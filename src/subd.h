#ifndef __SUBD_H__ 
#define __SUBD_H__ 

#include "config.h"

#include <string>
#include <vector>
#include <librta/librta.h>

#if HAVE_LIBOSDINTERFACE == 1
#include <rta-0.0.1/subdiv/osdi.h>
#endif

void add_subd_model(const std::string &filename, const std::string &displacement, const std::string &proxy, const std::string &pose, const std::string &spec, const std::string &occ);
void add_subd_obj_model(const std::string &filename, const std::string &proxy);
void load_subd_proxies();

#if HAVE_LIBOSDINTERFACE == 1
extern std::vector<OSDI::Model*> subd_models;
rta::rt_set* generate_compressed_bvhs_and_tracer(int w, int h);
#endif

#endif

