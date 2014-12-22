#include "tracers.h"
#include "material.h"
#include "util.h"

#include <librta/basic_types.h>
#include <bbvh-cuda/bbvh-cuda-node.h>

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <librta/intersect.h>

#define ray_x (blockIdx.x * blockDim.x + threadIdx.x)
#define ray_y (blockIdx.y * blockDim.y + threadIdx.y)

namespace rta {
	namespace cuda {
		namespace k {

			#define is_inner(F)       ((__float_as_int(F.x)&0x01)==1)
			#define extract_left(F)   (__float_as_int(F.x)>>1)
			#define extract_right(F)  (__float_as_int(F.y))
			#define extract_count(F)  (__float_as_int(F.x)>>1)
			#define extract_offset(F) (__float_as_int(F.y))
			#define box_min_x(nodes)  (nodes[0]).z
			#define box_min_y(nodes)  (nodes[0]).w
			#define box_min_z(nodes)  (nodes[1]).x
			#define box_max_x(nodes)  (nodes[1]).y
			#define box_max_y(nodes)  (nodes[1]).z
			#define box_max_z(nodes)  (nodes[1]).w

			__global__ void trace_cis_ailabox_indexed_with_alphamaps(cuda::simple_triangle *triangles, int n, float4 *nodes, uint *indices,
																	 vec3f *ray_orig, vec3f *ray_dir, float *max_t, int w, int h, 
																	 triangle_intersection<simple_triangle> *intersections, 
																	 cuda::material_t *mats, gi::cuda::mt_pool3f uniform_random_01) {
				if (ray_x < w && ray_y < h) {
					uint tid = ray_y*w+ray_x;
					// ray data
					vec3f orig = (ray_orig)[tid];
					vec3f dir = (ray_dir)[tid];
					float3 idir;
					float ooeps = exp2f(-80.0f); // Avoid div by zero.
					idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
					idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
					idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

					float t_max = max_t[tid];
					// stack mgmt
					uint32_t stack[32];
					int sp = -1;
					// cis info
					bool hit_left, hit_right;
					float dist_left = FLT_MAX, dist_right = FLT_MAX;;
					// triangle intersection
					triangle_intersection<simple_triangle> closest = intersections[tid];
					closest.t = FLT_MAX;
					float4 curr = nodes[0];

					while (true) {
						// fetch only first part, we don't need the box-only part.
						if (is_inner(curr)) {
							// load nodes
							float4 left_node[2], right_node[2];
							left_node[0] = nodes[2*extract_left(curr)+0];
							left_node[1] = nodes[2*extract_left(curr)+1];
							right_node[0] = nodes[2*extract_right(curr)+0];
							right_node[1] = nodes[2*extract_right(curr)+1];
							// extract data & intersect left
							float3 bb_min, bb_max;
							bb_min.x = box_min_x(left_node); bb_min.y = box_min_y(left_node); bb_min.z = box_min_z(left_node);
							bb_max.x = box_max_x(left_node); bb_max.y = box_max_y(left_node); bb_max.z = box_max_z(left_node);
							hit_left = intersect_aabb_aila(bb_min, bb_max, *(float3*)&orig, idir, closest.t, dist_left);
							// extract data & intersect right
							bb_min.x = box_min_x(right_node); bb_min.y = box_min_y(right_node); bb_min.z = box_min_z(right_node);
							bb_max.x = box_max_x(right_node); bb_max.y = box_max_y(right_node); bb_max.z = box_max_z(right_node);
							hit_right = intersect_aabb_aila(bb_min, bb_max, *(float3*)&orig, idir, closest.t, dist_right);
							if (dist_left >= closest.t || dist_left > t_max) hit_left = false;
							if (dist_right >= closest.t || dist_right > t_max) hit_right = false;
							// eval
							if (hit_left)
								if (hit_right) // note how we re-use the already loaded nodes.
									if (dist_left <= dist_right) {
										stack[++sp] = extract_right(curr);
										curr = left_node[0];
									}
									else {
										stack[++sp] = extract_left(curr);
										curr = right_node[0];
									}
								else // only hit the left node
									curr = left_node[0];
							else if (hit_right)
								curr = right_node[0];
							else if (sp >= 0)
								curr = nodes[2*stack[sp--]];
							else
								break;
						}
						else {
							uint elems = extract_count(curr);
							uint offset = extract_offset(curr);
							for (int i = 0; i < elems; ++i) {
								uint tri_id = indices[offset+i];
								triangle_intersection<simple_triangle> is(tri_id);
								simple_triangle tri = triangles[tri_id];
								if (intersect_tri_opt(tri, &orig, &dir, is)) {
									if (is.t < closest.t && is.t <= t_max) {
										// HERE's the difference: check for alpha maps
										material_t mat = mats[tri.material_index];
										float alpha = mat.alpha;
										if (mat.alpha_texture) {
											float3 bc;
											float2 T;
											is.barycentric_coord(&bc);
											barycentric_interpolation(&T, &bc, &tri.ta, &tri.tb, &tri.tc);
											float3 texel = mat.diffuse_texture->sample_nearest(T.x, T.y);
											alpha *= (texel.x + texel.y + texel.z)*0.33333;
										}
										float3 rnd = gi::next_random3f(uniform_random_01, ray_y*w+ray_x);
										bool use = (rnd.z <= alpha);
										if (use)
											closest = is;
									}
								}
							}
							if (sp >= 0)
								curr = nodes[2*stack[sp--]];
							else
								break;
						}
					}
					intersections[tid] = closest;
				}
			}
			

			#undef is_inner
			#undef extract_left
			#undef extract_right
			#undef extract_count
			#undef extract_offset
			#undef box_min_x
			#undef box_min_y
			#undef box_min_z
			#undef box_max_x
			#undef box_max_y
			#undef box_max_z
		}



		void trace_cis_ailabox_indexed_with_alphamaps(cuda::simple_triangle *triangles, int n, bbvh_node_float4<cuda::simple_aabb> *nodes, uint *indices,
													  vec3f *ray_orig, vec3f *ray_dir, float *max_t, int w, int h, 
													  triangle_intersection<cuda::simple_triangle> *is, 
													  cuda::material_t *materials, gi::cuda::mt_pool3f uniform_random_01) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = block_configuration_2d(w, h, threads);
			k::trace_cis_ailabox_indexed_with_alphamaps<<<blocks, threads>>>(triangles, n, (float4*)nodes, indices, 
																			 ray_orig, ray_dir, max_t, w, h, is, materials, uniform_random_01);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
		}
	}
}

