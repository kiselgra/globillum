#ifndef __CGLS_LIGHTS_H__ 
#define __CGLS_LIGHTS_H__ 

#include <libcgls/scene.h>

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

namespace rta {
	namespace cuda {
		namespace cgls {

			struct light {
				enum types { hemi, spot };
				types type;
				float3 dir, pos, col;
				float spot_cos_cutoff;
			};


			light* convert_and_upload_lights(scene_ref scene, int &N);
			void update_lights(scene_ref scene, light *data, int n);
		
			void add_shading(int w, int h, float3 *color_data, light *lights, int nr_of_lights, 
							 triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles);
		}
	}
}


#endif

