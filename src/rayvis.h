#ifndef __RAYVIS_H__ 
#define __RAYVIS_H__ 


void init_rayvis(int maxlen, int w, int h);
void restart_rayvis();
void add_vertex_to_all_rays(float3 v);
void render_rayvis();

#endif

