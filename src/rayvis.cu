#include "rayvis.h"

#include <libcgls/cgls.h>

#include <libhyb/trav-util.h>

#include <cuda_gl_interop.h>
#include <iostream>

using namespace std;
using namespace rta;
using namespace rta::cuda;

mesh_ref ray_mesh;
shader_ref ray_shader;
drawelement_ref ray;
int max_raylength;
int curr_length;
int rayvis_w, rayvis_h;


/*
 *	maxlen=4
 *
 *   0----1=2----3=4----5=6----7
 *
 *   add point x: (curr_v=0)
 *   x----x=x----x=x----x=x----x
 *
 *   add point y: (curr_v=1)
 *   x----y=y----y=y----y=y----y
 *
 *   add point z: (curr_v=2)
 *   x----y=y----z=z----z=z----z
 *
 */

void init_rayvis(int maxlen, int w, int h) {
	ray_mesh = make_mesh("rayvis", 1);
	float3 *data = new float3[w*h*2*maxlen];
	bind_mesh_to_gl(ray_mesh);
	set_mesh_primitive_type(ray_mesh, GL_LINES);
	add_vertex_buffer_to_mesh(ray_mesh, "vt", GL_FLOAT, w*h*2*maxlen, 3, data, GL_STATIC_DRAW);
	unbind_mesh_from_gl(ray_mesh);

	cudaGLRegisterBufferObject(mesh_vertex_buffer(ray_mesh, 0));
	rayvis_w = w;
	rayvis_h = h;
	max_raylength = maxlen;

	ray_shader = find_shader("rayvis");
	vec4f c(1,0,0,1);
	material_ref mat = make_material("rayvis", &c, &c, &c);
	ray = make_drawelement("rayvis", ray_mesh, ray_shader, mat);
	prepend_drawelement_uniform_handler(ray, (uniform_setter_t)default_material_uniform_handler);
	prepend_drawelement_uniform_handler(ray, (uniform_setter_t)default_matrix_uniform_handler);
}

void restart_rayvis() {
	curr_length = 0;
}

namespace k {
	__global__ void add_vertex_to_all_rays(int w, int h, float3 *data, float3 v, int curr_v, int N) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int offset = 2*N*(gid.y*w+gid.x);
		int i = 0;
		if (curr_v > 0) {
			i = 2*(curr_v-1)+1;
		}
		while (i < 2*N) {
			data[offset+i] = v;
			++i;
		}
	}

	// assumes w << src_w
	__global__ void add_intersections_to_rays(int w, int h, float3 *data, int curr_v, int N, int src_w, int src_h, 
											  triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;

		int stride_x = int(ceil(float(src_w)/float(w)));
		int stride_y = int(ceil(float(src_h)/float(h)));

		int src_x = gid.x * stride_x;
		int src_y = gid.y * stride_y;

		int src_id = src_y * src_w + src_x;
		triangle_intersection<cuda::simple_triangle> is = ti[src_id];
		if (!is.valid())
			return;
		float3 bc; 
		float3 P;
		cuda::simple_triangle tri = triangles[is.ref];
		is.barycentric_coord(&bc);
		barycentric_interpolation(&P, &bc, &tri.a, &tri.b, &tri.c);

		int offset = 2*N*(gid.y*w+gid.x);
		int i = 0;
		if (curr_v > 0) {
			i = 2*(curr_v-1)+1;
		}
		while (i < 2*N) {
			data[offset+i] = P;
			++i;
		}
	}
}

void add_vertex_to_all_rays(float3 v) {
	float3 *vertices;
	checked_cuda(cudaGLMapBufferObject((void**)&vertices, mesh_vertex_buffer(ray_mesh, 0)));
	
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(rayvis_w, rayvis_h, threads);
	k::add_vertex_to_all_rays<<<blocks, threads>>>(rayvis_w, rayvis_h, vertices, v, curr_length, max_raylength);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());

	++curr_length;

	checked_cuda(cudaGLUnmapBufferObject(mesh_vertex_buffer(ray_mesh, 0)));
}

void add_intersections_to_rays(int src_w, int src_h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles) {

	float3 *vertices;
	checked_cuda(cudaGLMapBufferObject((void**)&vertices, mesh_vertex_buffer(ray_mesh, 0)));
	
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(rayvis_w, rayvis_h, threads);
	k::add_intersections_to_rays<<<blocks, threads>>>(rayvis_w, rayvis_h, vertices, curr_length, max_raylength, src_w, src_h, ti, triangles);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());

	++curr_length;

	checked_cuda(cudaGLUnmapBufferObject(mesh_vertex_buffer(ray_mesh, 0)));
}


void render_rayvis() {
	render_drawelement(ray);
}

