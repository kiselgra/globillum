#ifndef __SUBD_H__ 
#define __SUBD_H__ 

#include <string>
#include <librta/librta.h>

void add_subd_model(const std::string &filename, const std::string &displacement);
rta::rt_set* generate_compressed_bvhs_and_tracer(int w, int h);

#endif

