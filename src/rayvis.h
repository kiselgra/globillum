#ifndef __RAYVIS_H__ 
#define __RAYVIS_H__ 

#include "util.h"
#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

void init_rayvis(int maxlen, int w, int h);
void restart_rayvis();
void add_vertex_to_all_rays(float3 v);
void add_intersections_to_rays(int src_w, int src_h, rta::triangle_intersection<rta::cuda::simple_triangle> *ti, rta::cuda::simple_triangle *triangles);
void render_rayvis();

#endif

