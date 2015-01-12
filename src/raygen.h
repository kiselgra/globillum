#ifndef __GI_RAYGEN_H__ 
#define __GI_RAYGEN_H__ 

#include "util.h"
#include "dofrays.h"

#ifndef __CUDACC__
#include <librta/raytrav.h>
#endif

namespace rta {
	namespace cuda {

		void setup_jittered_shirley(float *dirs, float *orgs, float *maxts, 
									float fovy, float aspect, int w, int h, float3 *view_dir, float3 *pos, float3 *up, float maxt,
									gi::cuda::mt_pool3f uniform_random_01);
	
		void setup_jittered_lens_shirley(float *dirs, float *orgs, float *maxts, 
										 float fovy, float aspect, int w, int h, float3 *view_dir, float3 *pos, float3 *up, float maxt,
										 float focus_distance, float aperture, float eye_to_lens,
										 gi::cuda::mt_pool3f uniform_random_01);
#ifndef __CUDACC__
		/*! \brief An extension of \ref rta::cuda::cam_ray_generator_shirley that
		 *  computes ray origins on the lens and adapts the directions so that all
		 *  rays generated for a given pixel will converge on the focal plane.
		 */
		class lens_ray_generator : public rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> {
		protected:
			gi::cuda::mt_pool3f jitter;
		public:
			float focus_distance, aperture, eye_to_lens;
			lens_ray_generator(uint res_x, uint res_y, 
							   float focus_distance, float aperture, float eye_to_lens,
							   gi::cuda::mt_pool3f jitter) 
				: rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(res_x, res_y),
				focus_distance(focus_distance), aperture(aperture), eye_to_lens(eye_to_lens),
				jitter(jitter) {
				}
			virtual void generate_rays() {
				rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>::generate_rays();
				rta::cuda::setup_shirley_lens_rays(this->gpu_direction, this->gpu_origin, this->gpu_maxt, 
												   fovy, aspect, this->w, this->h, (float3*)&dir, (float3*)&position, (float3*)&up, FLT_MAX,
												   focus_distance, aperture, eye_to_lens, jitter);
			}
			virtual std::string identification() { return "cuda version of the ray generator according to shirley."; }
			virtual void dont_forget_to_initialize_max_t() {}
		};


		/*! \brief An extension of \ref rta::cuda::cam_ray_generator_shirley that
		 *  samples rays on each pixel's footprint.
		 *  note that ray differentials are generated based on a uniform ray
		 *  distribution, see \ref rta::cam_ray_generator_shirley.
		 */
		class jittered_ray_generator : public rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> {
		protected:
			gi::cuda::mt_pool3f jitter;
		public:
			jittered_ray_generator(uint res_x, uint res_y, 
								   gi::cuda::mt_pool3f jitter) 
				: rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(res_x, res_y),
				  jitter(jitter) {
				}
			virtual void generate_rays() {
				rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>::generate_rays();
				rta::cuda::setup_jittered_shirley(this->gpu_direction, this->gpu_origin, this->gpu_maxt, 
												  fovy, aspect, this->w, this->h, (float3*)&dir, (float3*)&position, (float3*)&up, FLT_MAX,
												  jitter);
			}
			virtual std::string identification() { return "cuda version of the ray generator according to shirley."; }
			virtual void dont_forget_to_initialize_max_t() {}
		};
		
		/*! \brief An extension of \ref rta::cuda::cam_ray_generator_shirley that
		 *  computes ray origins on the lens and adapts the directions so that all
		 *  rays generated for a given pixel will converge on the focal plane.
		 */
		class jittered_lens_ray_generator : public rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials> {
		protected:
			gi::cuda::mt_pool3f jitter;
		public:
			float focus_distance, aperture, eye_to_lens;
			jittered_lens_ray_generator(uint res_x, uint res_y, 
										float focus_distance, float aperture, float eye_to_lens,
										gi::cuda::mt_pool3f jitter) 
				: rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>(res_x, res_y),
				focus_distance(focus_distance), aperture(aperture), eye_to_lens(eye_to_lens),
				jitter(jitter) {
				}
			virtual void generate_rays() {
				rta::cuda::camera_ray_generator_shirley<cuda::gpu_ray_generator_with_differentials>::generate_rays();
				rta::cuda::setup_jittered_lens_shirley(this->gpu_direction, this->gpu_origin, this->gpu_maxt, 
													   fovy, aspect, this->w, this->h, (float3*)&dir, (float3*)&position, (float3*)&up, FLT_MAX,
													   focus_distance, aperture, eye_to_lens, jitter);
			}
			virtual std::string identification() { return "cuda version of the ray generator according to shirley."; }
			virtual void dont_forget_to_initialize_max_t() {}
		};

#endif

	}
}



#endif

