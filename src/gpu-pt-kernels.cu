#include "gpu-pt-kernels.h"

#include "util.h"

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>
#include <libhyb/trav-util.h>

using namespace rta;
using namespace rta::cuda;

namespace k {
	__global__ void reset_data(float3 *data, uint w, uint h, float3 val) {
		int2 gid = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
							 blockIdx.y * blockDim.y + threadIdx.y);
		if (gid.x >= w || gid.y >= h) return;
		int id = gid.y*w+gid.x;
		data[id] = val;
	}
}

void reset_gpu_buffer(float3 *data, uint w, uint h, float3 val) {
	checked_cuda(cudaPeekAtLastError());
	dim3 threads(16, 16);
	dim3 blocks = block_configuration_2d(w, h, threads);
	k::reset_data<<<blocks, threads>>>(data, w, h, val);
	checked_cuda(cudaPeekAtLastError());
	checked_cuda(cudaDeviceSynchronize());
	}
