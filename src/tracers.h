#ifndef __GI_EXTRA_TRACERS_H__ 
#define __GI_EXTRA_TRACERS_H__ 

#ifndef __CUDACC__

#include <bbvh/bbvh.h>
#include <bbvh/binned-sah-bvh.h>
#include <bbvh-cuda/bbvh-cuda.h>

#include "material.h"
#include "util.h"

namespace rta {
	namespace cuda {
		//! Cuda CIS tracer using Aila et al's box hack.
		template<box_t__and__tri_t,	typename bvh_t>
		class bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas : public bbvh_gpu_cis_ailabox_indexed_tracer<forward_traits, bvh_t> {
			public:
				typedef bvh_t bbvh_t;
				typedef typename bbvh_t::node_t node_t;
				declare_traits_types;
				cuda::material_t *materials;
				gi::cuda::mt_pool3f uniform_random_01;
				bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas(rta::ray_generator *gen, bbvh_t *bvh, class bouncer *b, 
																   cuda::material_t *materials, gi::cuda::mt_pool3f pool)
				: bbvh_gpu_cis_ailabox_indexed_tracer<forward_traits, bvh_t>(gen, bvh, b), materials(materials), uniform_random_01(pool) {
				}
				bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas(const bbvh_gpu_cis_ailabox_indexed_shadow_tracer<forward_traits, bbvh_t> &sibling)
				: bbvh_gpu_cis_ailabox_indexed_tracer<forward_traits, bbvh_t>(sibling) {
				}
				virtual std::string identification() { return "cuda indexed bbvh ailabox tracer with support for alpha maps."; }
				virtual float trace_rays() {
					wall_time_timer wtt; wtt.start();
					void trace_cis_ailabox_indexed_with_alphamaps(tri_t *triangles, int n, node_t *nodes, uint *indices,
																  vec3f *ray_orig, vec3f *ray_dir, float *max_t, int w, int h, 
																  triangle_intersection<tri_t> *is, cuda::material_t *materials,
																  gi::cuda::mt_pool3f uniform_random_01);
					trace_cis_ailabox_indexed_with_alphamaps(this->bbvh->triangle_data.data, this->bbvh->triangle_data.n,
															 this->bbvh->node_data.data, this->bbvh->index_data.data,
															 (vec3f*)this->gpu_raygen->gpu_origin, (vec3f*)this->gpu_raygen->gpu_direction, 
															 this->gpu_raygen->gpu_maxt,
															 this->gpu_raygen->w, this->gpu_raygen->h,
															 this->gpu_bouncer->gpu_last_intersection, materials, uniform_random_01);
					float ms = wtt.look();
					return ms;
				}
				virtual bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas* copy() {
					return new bbvh_gpu_cis_ailabox_indexed_tracer_with_alphapmas(*this);
				}
// 				virtual bbvh_gpu_cis_ailabox_indexed_shadow_tracer<forward_traits, bvh_t>* matching_any_hit_tracer() {
// 					throw std::logic_error("there is no silbling tracer for indexed-bbvh-ailabox that computes any intersection. But it should be managable to add it.");
// 				}
		};
	}
}

#endif


#endif

