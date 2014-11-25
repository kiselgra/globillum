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

		// http://primes.utm.edu/lists/small/1000.txt
		static uint primes[500] = {
				2    ,  3    ,  5     , 7    , 11    , 13    , 17    , 19     ,23    , 29 ,
				31   ,  37   ,  41    , 43   ,  47   ,  53   ,  59   ,  61    , 67   ,  71 ,
				73   ,  79   ,  83    , 89   ,  97   , 101   , 103   , 107    ,109   , 113 ,
				127  ,  131  ,  137   , 139  ,  149  ,  151  ,  157  ,  163   , 167  ,  173 ,
				179  ,  181  ,  191   , 193  ,  197  ,  199  ,  211  ,  223   , 227  ,  229 ,
				233  ,  239  ,  241   , 251  ,  257  ,  263  ,  269  ,  271   , 277  ,  281 ,
				283  ,  293  ,  307   , 311  ,  313  ,  317  ,  331  ,  337   , 347  ,  349 ,
				353  ,  359  ,  367   , 373  ,  379  ,  383  ,  389  ,  397   , 401  ,  409 ,
				419  ,  421  ,  431   , 433  ,  439  ,  443  ,  449  ,  457   , 461  ,  463 ,
				467  ,  479  ,  487   , 491  ,  499  ,  503  ,  509  ,  521   , 523  ,  541 ,
				547  ,  557  ,  563   , 569  ,  571  ,  577  ,  587  ,  593   , 599  ,  601 ,
				607  ,  613  ,  617   , 619  ,  631  ,  641  ,  643  ,  647   , 653  ,  659 ,
				661  ,  673  ,  677   , 683  ,  691  ,  701  ,  709  ,  719   , 727  ,  733 ,
				739  ,  743  ,  751   , 757  ,  761  ,  769  ,  773  ,  787   , 797  ,  809 ,
				811  ,  821  ,  823   , 827  ,  829  ,  839  ,  853  ,  857   , 859  ,  863 ,
				877  ,  881  ,  883   , 887  ,  907  ,  911  ,  919  ,  929   , 937  ,  941 ,
				947  ,  953  ,  967   , 971  ,  977  ,  983  ,  991  ,  997   ,1009  , 1013 ,
				1019 ,  1021 ,  1031  , 1033 ,  1039 ,  1049 ,  1051 ,  1061  , 1063 ,  1069 ,
				1087 ,  1091 ,  1093  , 1097 ,  1103 ,  1109 ,  1117 ,  1123  , 1129 ,  1151 ,
				1153 ,  1163 ,  1171  , 1181 ,  1187 ,  1193 ,  1201 ,  1213  , 1217 ,  1223 ,
				1229 ,  1231 ,  1237  , 1249 ,  1259 ,  1277 ,  1279 ,  1283  , 1289 ,  1291 ,
				1297 ,  1301 ,  1303  , 1307 ,  1319 ,  1321 ,  1327 ,  1361  , 1367 ,  1373 ,
				1381 ,  1399 ,  1409  , 1423 ,  1427 ,  1429 ,  1433 ,  1439  , 1447 ,  1451 ,
				1453 ,  1459 ,  1471  , 1481 ,  1483 ,  1487 ,  1489 ,  1493  , 1499 ,  1511 ,
				1523 ,  1531 ,  1543  , 1549 ,  1553 ,  1559 ,  1567 ,  1571  , 1579 ,  1583 ,
				1597 ,  1601 ,  1607  , 1609 ,  1613 ,  1619 ,  1621 ,  1627  , 1637 ,  1657 ,
				1663 ,  1667 ,  1669  , 1693 ,  1697 ,  1699 ,  1709 ,  1721  , 1723 ,  1733 ,
				1741 ,  1747 ,  1753  , 1759 ,  1777 ,  1783 ,  1787 ,  1789  , 1801 ,  1811 ,
				1823 ,  1831 ,  1847  , 1861 ,  1867 ,  1871 ,  1873 ,  1877  , 1879 ,  1889 ,
				1901 ,  1907 ,  1913  , 1931 ,  1933 ,  1949 ,  1951 ,  1973  , 1979 ,  1987 ,
				1993 ,  1997 ,  1999  , 2003 ,  2011 ,  2017 ,  2027 ,  2029  , 2039 ,  2053 ,
				2063 ,  2069 ,  2081  , 2083 ,  2087 ,  2089 ,  2099 ,  2111  , 2113 ,  2129 ,
				2131 ,  2137 ,  2141  , 2143 ,  2153 ,  2161 ,  2179 ,  2203  , 2207 ,  2213 ,
				2221 ,  2237 ,  2239  , 2243 ,  2251 ,  2267 ,  2269 ,  2273  , 2281 ,  2287 ,
				2293 ,  2297 ,  2309  , 2311 ,  2333 ,  2339 ,  2341 ,  2347  , 2351 ,  2357 ,
				2371 ,  2377 ,  2381  , 2383 ,  2389 ,  2393 ,  2399 ,  2411  , 2417 ,  2423 ,
				2437 ,  2441 ,  2447  , 2459 ,  2467 ,  2473 ,  2477 ,  2503  , 2521 ,  2531 ,
				2539 ,  2543 ,  2549  , 2551 ,  2557 ,  2579 ,  2591 ,  2593  , 2609 ,  2617 ,
				2621 ,  2633 ,  2647  , 2657 ,  2659 ,  2663 ,  2671 ,  2677  , 2683 ,  2687 ,
				2689 ,  2693 ,  2699  , 2707 ,  2711 ,  2713 ,  2719 ,  2729  , 2731 ,  2741 ,
				2749 ,  2753 ,  2767  , 2777 ,  2789 ,  2791 ,  2797 ,  2801  , 2803 ,  2819 ,
				2833 ,  2837 ,  2843  , 2851 ,  2857 ,  2861 ,  2879 ,  2887  , 2897 ,  2903 ,
				2909 ,  2917 ,  2927  , 2939 ,  2953 ,  2957 ,  2963 ,  2969  , 2971 ,  2999 ,
				3001 ,  3011 ,  3019  , 3023 ,  3037 ,  3041 ,  3049 ,  3061  , 3067 ,  3079 ,
				3083 ,  3089 ,  3109  , 3119 ,  3121 ,  3137 ,  3163 ,  3167  , 3169 ,  3181 ,
				3187 ,  3191 ,  3203  , 3209 ,  3217 ,  3221 ,  3229 ,  3251  , 3253 ,  3257 ,
				3259 ,  3271 ,  3299  , 3301 ,  3307 ,  3313 ,  3319 ,  3323  , 3329 ,  3331 ,
				3343 ,  3347 ,  3359  , 3361 ,  3371 ,  3373 ,  3389 ,  3391  , 3407 ,  3413 ,
				3433 ,  3449 ,  3457  , 3461 ,  3463 ,  3467 ,  3469 ,  3491  , 3499 ,  3511 ,
		};

		void compute_next_halton_batch(int w, int h, int b0, int b1, int b2, float3 *data) {
			checked_cuda(cudaDeviceSynchronize());
// 			checked_cuda(cudaPeekAtLastError());
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
