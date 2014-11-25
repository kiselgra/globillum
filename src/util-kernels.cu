#include "util.h"

#include <libcgl/wall-time.h>
#include <curand_kernel.h>

#include <iostream>

using namespace std;

namespace gi {
	namespace cuda {
		namespace k {

			__global__ void compute_halton_batch(int w, int h, float3 *data, uint b0, uint b1, uint b2) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				int id = gid.y*w+gid.x;
#define HALTON_VER 1
#if HALTON_VER == 1
				uint o = (b0+b1+b2)/3;
				float3 result = make_float3(0.0f, 0.0f, 0.0f);
				uint base = b0;
				float f = 1.0f / float(base);
				float f2 = f;
				uint i = id+o;

				while (i > 0) {
					result.x += f * (i%base);
					i = i / base;
					f *= f2;
				}

				base = b1;
				f = 1.0f / float(base);
				f2 = f;
				i = id+o;
				while (i > 0) {
					result.y += f * (i%base);
					i = i / base;
					f *= f2;
				}

				base = b2;
				f = 1.0f / float(base);
				f2 = f;
				i = id+o;
				while (i > 0) {
					result.z += f * (i%base);
					i = i / base;
					f *= f2;
				}
				data[id] = result;
#elif HALTON_VER == 2
				float3 r2 = make_float3(0,0,0);
				int o = 20;
				uint base_0 = b0;
				uint base_1 = b1;
				uint base_2 = b2;
				uint i_0 = id+o;
				uint i_1 = id+o;
				uint i_2 = id+o;
				float f_0 = 1.0f/float(base_0);
				float m_0 = f_0;
				float f_1 = 1.0f/float(base_1);
				float m_1 = f_1;
				float f_2 = 1.0f/float(base_2);
				float m_2 = f_2;
				while (i_0 > 0) {
					r2.x += f_0 * (i_0 % base_0);
					i_0 /= base_0;
					f_0 *= m_0;
					r2.y += f_1 * (i_1 % base_1);
					i_1 /= base_1;
					f_1 *= m_1;
					r2.z += f_2 * (i_2 % base_2);
					i_2 /= base_2;
					f_2 *= m_2;
				}
				data[id] = r2;
#elif HALTON_VER == 3
				float3 correct;
				{
					uint o = 20;
					float3 result = make_float3(0.0f, 0.0f, 0.0f);
					uint base = b0;
					float f = 1.0f / float(base);
					float f2 = f;
					uint i = id+o;

					while (i > 0) {
						result.x += f * (i%base);
						i = i / base;
						f *= f2;
					}

					base = b1;
					f = 1.0f / float(base);
					f2 = f;
					i = id+o;
					while (i > 0) {
						result.y += f * (i%base);
						i = i / base;
						f *= f2;
					}

					base = b2;
					f = 1.0f / float(base);
					f2 = f;
					i = id+o;
					while (i > 0) {
						result.z += f * (i%base);
						i = i / base;
						f *= f2;
					}
					correct = result;
				}
				uint o = 20;
				float3 result = make_float3(0.0f, 0.0f, 0.0f);
				uint base = b0;
				float f = 1.0f / float(base);
				float f2 = f;
				uint i = id+o;

				while (i > base) {
					result.x += f * (i%base);
					i = i / base;
					f *= f2;
					result.x += f * (i%base);
					i = i / base;
					f *= f2;
				}
				if (i > 0) {
					result.x += f * (i%base);
					i = i / base;
					f *= f2;
				}

				base = b1;
				f = 1.0f / float(base);
				f2 = f;
				i = id+o;
				while (i > base) {
					result.y += f * (i%base);
					i = i / base;
					f *= f2;
					result.y += f * (i%base);
					i = i / base;
					f *= f2;
				}
				if (i > 0) {
					result.y += f * (i%base);
					i = i / base;
					f *= f2;
				}

				base = b2;
				f = 1.0f / float(base);
				f2 = f;
				i = id+o;
				while (i > base) {
					result.z += f * (i%base);
					i = i / base;
					f *= f2;
					result.z += f * (i%base);
					i = i / base;
					f *= f2;
				}
				if (i > 0) {
					result.z += f * (i%base);
					i = i / base;
					f *= f2;
				}
				data[id] = result;
				if (result.x != correct.x || result.y != correct.y || result.z != correct.z)
					printf("BAAAA\n");
#endif
			}
		}

		/*
		void compute_next_halton_batch(int w, int h, int b0, int b1, int b2, float3 *data) {
			checked_cuda(cudaDeviceSynchronize());
			wall_time_t t0 = wall_time_in_ms();
			dim3 threads(16, 16);
			dim3 blocks = rta::cuda::block_configuration_2d(w, h, threads);
			k::compute_halton_batch<<<blocks, threads>>>(w, h, data, b0, b1, b2);
			checked_cuda(cudaDeviceSynchronize());
			const int N = 8;
			int off=800*300;
			float3 test[N];
			cudaMemcpy(test, data+off, N*sizeof(float3), cudaMemcpyDeviceToHost);
			for (int i = 0; i < N; ++i)
				printf("[%d] %6.6f %6.6f %6.6f\n", i, test[i].x, test[i].y, test[i].z);
			wall_time_t t1 = wall_time_in_ms();
			cout << "computing a batch on the gpu took " << t1-t0 << " ms. (" << b0 << ", " << b1 << ", " << b2 << ")" << endl;
		}
		
		void update_halton_pool(halton_pool3f hp, int batch_nr) {
			uint b0 = primes[3*batch_nr + hp.prime_offset + 0];
			uint b1 = primes[3*batch_nr + hp.prime_offset + 1];
			uint b2 = primes[3*batch_nr + hp.prime_offset + 2];
			compute_next_halton_batch(hp.w, hp.h, b0, b1, b2, hp.data);
		}
		*/

		// 
		// MT
		//

		namespace k {
			__global__ void initialize_mt(int w, int h, curandStateMRG32k3a *state) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				int id = gid.y*w+gid.x;
				/* Each thread gets same seed, a different sequence 
				   number, no offset */
				curand_init(0, id, 0, &state[id]);
			}

			__global__ void update_mt_uniform(int w, int h, curandStateMRG32k3a *state, float3 *result) {
				int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
									 blockIdx.y * blockDim.y + threadIdx.y);
				if (gid.x >= w || gid.y >= h) return;
				int id = gid.y*w+gid.x;
				/* Copy state to local memory for efficiency */
				curandStateMRG32k3a localState = state[id];
				/* Generate pseudo-random uniforms */
				float3 f = make_float3(curand_uniform(&localState),
									   curand_uniform(&localState),
									   curand_uniform(&localState));
				/* Copy state back to global memory */
				state[id] = localState;
				/* Store results */
				result[id] = f;
			}
		}

		mt_pool3f generate_mt_pool_on_gpu(int w, int h) {
			mt_pool3f pool;
			checked_cuda(cudaMalloc(&pool.data, sizeof(float3)*w*h));
			checked_cuda(cudaMalloc(&pool.mt_states, w * h * sizeof(curandStateMRG32k3a)));
			pool.w = w;
			pool.h = h;

			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = rta::cuda::block_configuration_2d(w, h, threads);
			k::initialize_mt<<<blocks, threads>>>(w, h, (curandStateMRG32k3a*)pool.mt_states);
			checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
			return pool;
		}

		void update_mt_pool(mt_pool3f mp) {
			checked_cuda(cudaPeekAtLastError());
			dim3 threads(16, 16);
			dim3 blocks = rta::cuda::block_configuration_2d(mp.w, mp.h, threads);
			checked_cuda(cudaDeviceSynchronize());
			wall_time_t t0 = wall_time_in_ms();
			k::update_mt_uniform<<<blocks, threads>>>(mp.w, mp.h, (curandStateMRG32k3a*)mp.mt_states, mp.data);
			// 	checked_cuda(cudaPeekAtLastError());
			checked_cuda(cudaDeviceSynchronize());
			wall_time_t t1 = wall_time_in_ms();
			printf("computing a batch of mt numbers took %6.6f ms\n", t1-t0);

		}

	}
}
