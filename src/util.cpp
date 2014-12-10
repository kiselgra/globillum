#include "util.h"

#include <omp.h>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <png++/png.hpp>

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

	std::string image_store_path = "/tmp/";

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

		/*
		halton_pool3f generate_halton_pool_on_gpu(int w, int h, int offset) {
			halton_pool3f pool;
			pool.N = w*h;
			checked_cuda(cudaMalloc(&pool.data, sizeof(float3)*pool.N));
			pool.w = w;
			pool.h = h;
			pool.prime_offset = offset;
			return pool;
		}
		*/

		lcg_random_state generate_lcg_pool_on_gpu(int N) {
			lcg_random_state pool;
			pool.N = N;
			unsigned int *host = new unsigned int[N];
			#pragma omp parallel private(seed)
			{
				unsigned int seed = omp_get_thread_num();
				//#pragma omp for schedule(dynamic, 32) private(seed)
				#pragma omp for
				for (int i = 0; i < N; ++i) {
					host[i] = rand_r(&seed);
				}
			}
			checked_cuda(cudaMalloc(&pool.data, sizeof(unsigned int)*N));
			checked_cuda(cudaMemcpy(pool.data, host, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));
			delete [] host;
			return pool;
		}
		
		void download_and_save_image(const std::string &basename, int seq, int w, int h, float3 *color) {
			float3 *data = new float3[w*h];
			checked_cuda(cudaMemcpy(data, color, w*h*sizeof(float3), cudaMemcpyDeviceToHost));
			checked_cuda(cudaDeviceSynchronize());
			ostringstream oss; oss << image_store_path << basename << "." << setw(4) << setfill('0') << right << seq;

			png::image<png::rgb_pixel> image(w, h);
			for (int y = 0; y < h; ++y) {
				int y_out = h - y - 1;
				for (int x = 0; x < w; ++x) {
					float3 *pixel = data+y*w+x;
					image.set_pixel(w-x-1, y_out, png::rgb_pixel(clamp(int(255*pixel->x),0,255), 
																 clamp(int(255*pixel->y),0,255), clamp(int(255*pixel->z),0,255))); 
				}
			}

			image.write(oss.str() + ".png");
		}
	}
}


/* vim: set foldmethod=marker: */

