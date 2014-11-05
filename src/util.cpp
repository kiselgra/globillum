#include "util.h"

using namespace std;

namespace gi {

	float halton(int index, int base) {
		float result = 0.0f;
		float f = 1.0f / float(base);
		int i = index;
		while (i > 0) {
			result += f * (i%base);
			i = i / base;
			f = f / float(base);
		}
		return result;
	}

	namespace cuda {

		halton_pool2f generate_halton_pool_on_gpu(int N) {
			halton_pool2f pool;
			pool.N = N;
			float2 *host = new float2[N];
			#pragma omp parallel for schedule(dynamic, 32)
			for (int i = 0; i < N; ++i) {
				host[i].x = halton(i+1, 2);
				host[i].y = halton(i+1, 3);
			}
			checked_cuda(cudaMalloc(&pool.data, sizeof(float2)*N));
			checked_cuda(cudaMemcpy(pool.data, host, sizeof(float2)*N, cudaMemcpyHostToDevice));
			delete [] host;
			return pool;
		}

	}
}


/* vim: set foldmethod=marker: */

