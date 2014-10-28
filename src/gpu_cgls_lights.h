#ifndef __GPU_CGLS_LIGHTS_H__ 
#define __GPU_CGLS_LIGHTS_H__ 

#include "gi_algorithm.h"

namespace local {

	class gpu_cgls_lights : public gi_algorithm {
		int w, h;
	public:
		gpu_cgls_lights(int w, int h, const std::string &name = "gpu_cgls_lights") : gi_algorithm(name), w(w), h(h) {
		}
	};

}


#endif

