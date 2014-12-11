#ifndef __GI_LIGHTS_H__ 
#define __GI_LIGHTS_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

#include <vector>

namespace gi {
	struct rect_light {
		float3 center, dir, up, col;
		float2 wh;
	};

	struct light {
		enum type_t { rect };
		type_t type;
		float power;	//!< for light selection, only.

		union {
			struct rect_light rectlight;
		};
	};

	extern std::vector<light> lights;

	namespace cuda {

		light* convert_and_upload_lights(int &N, float &power);
		void update_lights(light *data, int N, float &power);

	}
}

#endif

