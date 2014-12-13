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

	struct sky_light {
		float scale;
		char *map;
		float3 *data;
		int w, h;
	};

	struct light {
		enum type_t { rect, sky };
		type_t type;
		float power;	//!< for light selection, only.

		union {
			struct rect_light rectlight;
			struct sky_light  skylight;
		};
	};

	extern std::vector<light> lights;

	namespace cuda {

		light* convert_and_upload_lights(int &N, float &power);
		void update_lights(light *data, int N, float &power);

	}
}

#endif

