#include "gpu_cgls_lights.h"

#include "material.h"
#include "vars.h"

#include <libhyb/trav-util.h>

using namespace std;
using namespace rta;

extern cuda::material_t *gpu_materials;

namespace local {

// 	vec3f *material;

	template<typename _box_t, typename _tri_t> struct gpu_material_evaluator : public cuda::gpu_ray_bouncer<forward_traits> {
		declare_traits_types;
		cuda::material_t *materials;
		cuda::simple_triangle *tri_ptr;
		float3 *material_colors;
		cuda::cam_ray_generator_shirley *crgs;
		gpu_material_evaluator(uint w, uint h, cuda::material_t *materials, cuda::simple_triangle *triangles, cuda::cam_ray_generator_shirley *crgs)
			: cuda::gpu_ray_bouncer<forward_traits>(w, h), materials(materials), material_colors(0), tri_ptr(triangles),
			  crgs(crgs) {
			checked_cuda(cudaMalloc(&material_colors, sizeof(float3)*w*h));
		}
		~gpu_material_evaluator() {
			checked_cuda(cudaFree(material_colors));
		}
		virtual bool trace_further_bounces() {
			return false;
		}
		virtual void bounce() {
			float dy = (2.0f/this->h) * tanf(crgs->fovy*0.5f*M_PI/180.0f);
			float2 diff = make_float2(crgs->aspect * dy, dy);
			rta::cuda::evaluate_material(this->w, this->h, this->gpu_last_intersection, tri_ptr, materials, material_colors, diff, crgs->gpu_origin, crgs->gpu_direction);
		}
		virtual std::string identification() {
			return "cuda primary intersection collector.";
		}
	};

	void gpu_cgls_lights::activate(rt_set *orig_set) {
		set = *orig_set;
		set.rt = set.rt->copy();
// 		set.bouncer = collector = new cuda::primary_intersection_collector<B,T>(w, h);
		cuda::simple_triangle *triangles = set.basic_as<B, T>()->triangle_ptr();
		set.rgen = crgs = new cuda::cam_ray_generator_shirley(w, h);
		set.bouncer = new gpu_material_evaluator<B, T>(w, h, gpu_materials, triangles, crgs);
// 		set.bouncer = collector = new cuda::primary_intersection_downloader<B,T, primary_intersection_collector<B,T>>(w, h);
		set.basic_rt<B, T>()->ray_bouncer(set.bouncer);
		set.basic_rt<B, T>()->ray_generator(set.rgen);
		cout << "-----BAS: " << set.as->identification() << endl;
		cout << "-----tri: " << set.basic_as<B, T>()->triangle_ptr() << endl;
		declare_variable<int>("lod", 0);
// 		material = new vec3f[w*h];
	}

	void gpu_cgls_lights::compute() {
			cout << "tracing..." << endl;
			vec3f pos, dir, up;
			matrix4x4f *lookat_matrix = lookat_matrix_of_cam(current_camera());
			extract_pos_vec3f_of_matrix(&pos, lookat_matrix);
			extract_dir_vec3f_of_matrix(&dir, lookat_matrix);
			extract_up_vec3f_of_matrix(&up, lookat_matrix);
			crgs->setup(&pos, &dir, &up, 2*camera_fovy(current_camera()));
			set.rt->trace();

			/*
			cuda::primary_intersection_downloader<B,T, primary_intersection_collector<B,T>> *C = (cuda::primary_intersection_downloader<B,T, primary_intersection_collector<B,T>>*)collector;

			T *triangles = set.basic_as<B, T>()->canonical_triangle_ptr();
			T::vec3_t bc, tmp;
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x) {
					const triangle_intersection<T> &ti = C->intersection(x,y);
					if (ti.valid()) {
						T &tri = triangles[ti.ref];
						ti.barycentric_coord(&bc);
						const T::vec3_t &va = vertex_a(tri);
						const T::vec3_t &vb = vertex_b(tri);
						const T::vec3_t &vc = vertex_c(tri);
						barycentric_interpolation(&tmp, &bc, &va, &vb, &vc);
						hitpoints.pixel(x,y) = { tmp.x, tmp.y, tmp.z };
						const T::vec3_t &na = normal_a(tri);
						const T::vec3_t &nb = normal_b(tri);
						const T::vec3_t &nc = normal_c(tri);
						barycentric_interpolation(&tmp, &bc, &na, &nb, &nc);
						normals.pixel(x,y) = { tmp.x, tmp.y, tmp.z };
					}
					else {
						make_vec3f(&hitpoints.pixel(x,y), FLT_MAX, FLT_MAX, FLT_MAX);
						make_vec3f(&normals.pixel(x,y), 0, 0, 0);
					}
				}
			set.basic_as<B, T>()->free_canonical_triangles(triangles);
			*/
			cout << "primary visibility took " << set.basic_rt<B, T>()->timings.front() << "ms." << endl;
	
			cout << "saving output" << endl;
			png::image<png::rgb_pixel> image(w, h);
			vec3f *data = new vec3f[w*h];
			cudaMemcpy(data, (dynamic_cast<gpu_material_evaluator<B, T>*>(set.bouncer))->material_colors, sizeof(float3)*w*h, cudaMemcpyDeviceToHost);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					vec3f *pixel = data+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(255*pixel->x,0,255), clamp(255*pixel->y,0,255), clamp(255*pixel->z,0,255))); 
				}
			}
			image.write("out.png");

// 			evaluate_material();
// 			save(material);
	}
		
	/*
	void gpu_cgls_lights::evaluate_material() {
			cout << "evaluating material..." << endl;
			cuda::primary_intersection_downloader<B,T, primary_intersection_collector<B,T>> *C = (cuda::primary_intersection_downloader<B,T, primary_intersection_collector<B,T>>*)collector;
			T *triangles = set.basic_as<B, T>()->canonical_triangle_ptr();
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					const triangle_intersection<T> &ti = C->intersection(x,y);
					if (ti.valid()) {
						T &tri = triangles[ti.ref];
						material_t *mat = rta::material(tri.material_index);
						vec3f col = (*mat)();
						if (mat->diffuse_texture) {
							T::vec3_t bc; 
							ti.barycentric_coord(&bc);
							const vector_traits<T::vec3_t>::vec2_t &ta = texcoord_a(tri);
							const vector_traits<T::vec3_t>::vec2_t &tb = texcoord_b(tri);
							const vector_traits<T::vec3_t>::vec2_t &tc = texcoord_c(tri);
							vector_traits<T::vec3_t>::vec2_t T;
							barycentric_interpolation(&T, &bc, &ta, &tb, &tc);
							vec2f t = { T.x, T.y };
							col = (*mat)(t);
						}
						material[y*w+x] = col;
					}
					else
						make_vec3f(material+y*w+x, 0, 0, 0);
				}
			}
			set.basic_as<B, T>()->free_canonical_triangles(triangles);
		}
		*/


	/*
		void gpu_cgls_lights::save(vec3f *out) {
			cout << "saving output" << endl;
			png::image<png::rgb_pixel> image(w, h);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					vec3f *pixel = out+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(255*pixel->x,0,255), clamp(255*pixel->y,0,255), clamp(255*pixel->z,0,255))); 
				}
			}
			image.write("out.png");
		}
		*/

}

/* vim: set foldmethod=marker: */

