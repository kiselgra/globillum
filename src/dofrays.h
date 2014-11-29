#ifndef __GI_DOFRAYS_H__ 
#define __GI_DOFRAYS_H__ 

#include "util.h"

namespace rta {
	namespace cuda {

		void setup_shirley_lens_rays(float *dirs, float *orgs, float *maxts, 
									 float fovy, float aspect, int w, int h, float3 *view_dir, float3 *pos, float3 *up, float maxt,
									 float focus_distance, float aperture, float eye_to_lens, gi::cuda::mt_pool3f uniform_random_01);

	}
}

#endif

