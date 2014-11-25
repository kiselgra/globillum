#ifndef __GI_UTIL_H__ 
#define __GI_UTIL_H__ 

/*! \file util.h
 * 	\brief general utilities for gi algorithms.
 * 	\note will also be included in cuda source. take care.
 */

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace gi {

	float halton(int index, int base);

	namespace cuda {
		
		/*! \brief 32 bit linear congruential random number generator, copied from gpu gems 3. */
		heterogenous inline unsigned int lcg_step(unsigned int *z, const unsigned int A, const unsigned int C)  {  
			return *z=(A**z+C);  
		}

		/*! \brief Random data for path tracing.
		 *
		 *  We store 3 floats for light sampling (choose light, choose position
		 *  on light) or path sampling (choose component, choose direction).
		 *  N gives the number of explicitely stored path samples.
		 */
		struct multi_bounce_halton_pool3f {
			float3 *data;
			int bounces;
			int chunksize;
			multi_bounce_halton_pool3f() : data(0), bounces(0), chunksize(0) {}
		};

		multi_bounce_halton_pool3f generate_multi_bounce_halton_pool_on_gpu(int N, int bounces, int b0, int b1, int b2);

		/*! \brief State for linear congruential random number generator.
		 *  
		 *  This is the cheapest random number generator, memory-wise.
		 */
		struct lcg_random_state {
			unsigned int *data;
			int N;
			lcg_random_state() : data(0), N(0) {}
			heterogenous float random_float(int id) {
				return 2.3283064365387e-10 * lcg_step(&data[id], 1664525, 1013904223U);
			}
		};

		/*! \brief Generated N random state cells to be used with the \ref
		 *  uniform_random_lcg generator.
		 *  \note The data is initialized.
		 *  \note The pointer in the returned structure points to gpu memory.
		 */
		lcg_random_state generate_lcg_pool_on_gpu(int N);

		
		
		struct halton_pool2f {
			float2 *data;
			int N;
			halton_pool2f() : data(0), N(0) {}
		};

		/*! \brief Generates N float2 values generated by halton (2,3). 
		 *  \note The data is initialized.
		 *  \note The pointer in the returned structure points to gpu memory.
		 */
		halton_pool2f generate_halton_pool_on_gpu(int N);
		
		
		
		struct halton_pool3f {
			float3 *data;
			int N;
			int w, h, prime_offset;
			halton_pool3f() : data(0), N(0), w(0), h(0), prime_offset(0) {}
		};
		
		/*! \brief Generates w*h float3 values. 
		 *  \note The data is *not* initialized.
		 *  \note The pointer in the returned structure points to gpu memory.
		 */
		halton_pool3f generate_halton_pool_on_gpu(int w, int h, int offset);
		/*! \brief Compute next batch of halton numbers and store them in the halton pool.
		 */
		void update_halton_pool(halton_pool3f hp, int batch_nr);



		struct mt_pool3f {
			float3 *data;
			int w, h;
			void *mt_states;	// right now i don't want to splash curand around...
			mt_pool3f() : data(0), w(0), h(0), mt_states(0) {}
		};
		
		/*! \brief Allocate w*h float3 values and initialize curand's mersenne twister.
		 *  \note The data is *not* initialized.
		 *  \note The pointer in the returned structure points to gpu memory.
		 */
		mt_pool3f generate_mt_pool_on_gpu(int w, int h);
		/*! \brief Compute next batch of mt numbers and store them in the pool.
		 */
		void update_mt_pool(mt_pool3f mp);




		struct random_sampler_path_info {
			int curr_path, max_paths;
			int curr_bounce, max_bounces;
		};

		heterogenous inline float3 next_random3f(halton_pool2f &pool, int id, const random_sampler_path_info &state) {
			float2 f2 = pool.data[(state.max_paths*state.max_bounces * (state.max_bounces*state.curr_path+state.curr_bounce) + id) % pool.N];
			return make_float3(f2.x, f2.y, f2.y);
		}

		heterogenous inline float3 next_random3f(lcg_random_state &pool, int id, const random_sampler_path_info &) {
			float3 f3;
			f3.x = pool.random_float(id);
			f3.y = pool.random_float(id);
			f3.z = pool.random_float(id);
			return f3;
		}
		
		heterogenous inline float3 next_random3f(multi_bounce_halton_pool3f &pool, int id, const random_sampler_path_info &state) {
			float3 f3 = pool.data[id];
			return f3;
		}

		heterogenous inline float3 next_random3f(halton_pool3f &pool, int id, const random_sampler_path_info &state) {
			float3 f3 = pool.data[id];
			return f3;
		}

		heterogenous inline float3 next_random3f(mt_pool3f &pool, int id, const random_sampler_path_info &state) {
			float3 f3 = pool.data[id];
			return f3;
		}


	}
}

#endif

