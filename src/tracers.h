#ifndef __GI_EXTRA_TRACERS_H__ 
#define __GI_EXTRA_TRACERS_H__ 

#ifndef __CUDACC__

#include <vector>
#include <iostream>

#include <bbvh/bbvh.h>
#include <bbvh/binned-sah-bvh.h>
#include <bbvh-cuda/bbvh-cuda.h>

#include "material.h"
#include "util.h"

namespace rta {

	template<typename _box_t, typename _tri_t> struct tandem_tracer : public rta::raytracer {
		declare_traits_types;
		
		typedef rta::basic_raytracer<box_t, tri_t> tracer_t;
		tracer_t *closest_hit_tracer;
		tracer_t *any_hit_tracer;
		tracer_t *use_tracer, *other, *last;

		tandem_tracer(tracer_t *ch, tracer_t *ah) : closest_hit_tracer(ch), any_hit_tracer(ah), use_tracer(0), other(0), last(0) {
		}
		void select_closest_hit_tracer() {
			use_tracer = closest_hit_tracer;
			other = any_hit_tracer;
		}
		void select_any_hit_tracer() {
			use_tracer = any_hit_tracer;
			other = closest_hit_tracer;
		}
		virtual void trace() {
			throw std::logic_error("a tandem tracer is for progressive tracing, only.");
		}
		// bounce() might change the tracer, or might keep it
		// therefore, after bounce() and tracer_furhter_boucnes() was called with the `last'
		// tracer, we copy its information over to both versions.
		virtual void trace_progressively(bool first) {
			last = use_tracer;
			use_tracer->trace_progressively(first);
			use_tracer->copy_progressive_state(last);
			other->copy_progressive_state(last);
		}
		virtual std::string identification() {
			return std::string("wrapper to trace using two basic_raytracers in tandem, in this case: (")
				   + closest_hit_tracer->identification() + ", and " + any_hit_tracer->identification() + ")";
		}
		virtual bool progressive_trace_running() {
			return use_tracer->progressive_trace_running();
		}
		virtual rta::raytracer* copy() {
			return new tandem_tracer(*this);
		}
		virtual bool supports_max_t() {
			return closest_hit_tracer->supports_max_t() && any_hit_tracer->supports_max_t();
		}
	};


	template<typename _box_t, typename _tri_t, typename tracer_t> 
	struct iterated_tracers : public rta::basic_raytracer<forward_traits> {
		declare_traits_types;
		
		tracer_t *first_tracer;
		std::vector<tracer_t*> other_tracers;

		iterated_tracers(tracer_t *first) 
		: rta::basic_raytracer<forward_traits>(first->raygen,first->bouncer,first->accel_struct), 
		  first_tracer(first) {
		}
		virtual std::string identification() {
			return std::string("wrapper to have a single ray tracer that calls trace_rays() "
							   "on a set of tracers, copying t_max values inbetween.");
		}
		virtual float copy_intersection_distance_to_max_t() = 0;
		virtual float trace_rays() {
			float sum = 0;
			sum += first_tracer->trace_rays();
			int n = other_tracers.size();
			for (int i = 0; i < n; ++i) {
				sum += copy_intersection_distance_to_max_t();
				sum += other_tracers[i]->trace_rays();
			}
		}
		void append_tracer(tracer_t *add) {
			other_tracers.push_back(add);
		}
		//! all connected tracers *must absolutely* support max_t.
		virtual bool supports_max_t() {
			return true;
		}
	};

	namespace cuda {
		
		void copy_intersection_distance_to_max_t(int w, int h, 
												 rta::triangle_intersection<rta::cuda::simple_triangle> *is, float *max_t);

		template<typename _box_t, typename _tri_t, typename sibling_t> 
		struct iterated_gpu_tracers : public iterated_tracers<forward_traits, rta::cuda::gpu_raytracer<forward_traits, sibling_t>> {
			declare_traits_types;
			
			typedef rta::cuda::gpu_raytracer<box_t, tri_t, sibling_t> tracer_t;

			iterated_gpu_tracers(tracer_t *first) 
			: iterated_tracers<forward_traits, rta::cuda::gpu_raytracer<forward_traits, sibling_t>>(first) {
			}
			virtual std::string identification() {
				return std::string("wrapper to have a single ray tracer that calls trace_rays() "
								   "on a set of GPU tracers, copying t_max values inbetween.");
			}
			virtual void prepare_trace() {
				reset_intersections(this->first_tracer->gpu_bouncer->gpu_last_intersection,
									this->first_tracer->gpu_bouncer->w, this->first_tracer->gpu_bouncer->h);
			}
			float copy_intersection_distance_to_max_t() {
				wall_time_timer wtt; wtt.start();
				rta::cuda::gpu_ray_bouncer<forward_traits> *bouncer = this->first_tracer->gpu_bouncer;
				rta::cuda::gpu_ray_generator *raygen = this->first_tracer->gpu_raygen;
				rta::cuda::copy_intersection_distance_to_max_t(raygen->w, raygen->h, bouncer->gpu_last_intersection, raygen->gpu_maxt);
				return wtt.look();
			}
			virtual rta::basic_raytracer<forward_traits>* copy() {
				return new iterated_gpu_tracers(*this);
			}
		};

	}
}

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

