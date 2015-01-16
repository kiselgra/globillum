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

#ifndef __CUDACC__

	/*! \brief An extension of \ref rta::cuda::cam_ray_generator_shirley that
	 *  computes ray origins on the lens and adapts the directions so that all
	 *  rays generated for a given pixel will converge on the focal plane.
	 */
	class jittered_lens_ray_generator : public rta::camera_ray_generator_shirley<ray_generator_with_differentials> {
	protected:
		float3 *uniform_random_01;
	public:
		float focus_distance, aperture, eye_to_lens;
		jittered_lens_ray_generator(uint res_x, uint res_y, 
									float focus_distance, float aperture, float eye_to_lens,
									float3 *jitter) 
			: rta::camera_ray_generator_shirley<ray_generator_with_differentials>(res_x, res_y),
			focus_distance(focus_distance), aperture(aperture), eye_to_lens(eye_to_lens),
			uniform_random_01(jitter) {
			}
		virtual void generate_rays() {
			// this takes care of setting up `proper' ray differentials
			rta::camera_ray_generator_shirley<rta::ray_generator_with_differentials>::generate_rays();

			int w = this->raydata.w, h = this->raydata.h;
			vec3f view_dir = this->dir;
			vec3f cam_pos = this->position;
			vec3f cam_up = this->up;

			#pragma omp parallel
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x) {

					int id = y * w + x;
					this->max_t(x,y) = FLT_MAX;
					float fovy = this->fovy / 2.0;
					float height = tanf(M_PI * fovy / 180.0f);
					float width = this->aspect * height;

					float3 random = gi::next_random3f(uniform_random_01, id);
					float jx = float(x) + random.x;
					float jy = float(y) + random.y;

					float u_s = ((jx)/(float)w) * 2.0f - 1.0f;	// \in (-1,1)
					float v_s = ((jy)/(float)h) * 2.0f - 1.0f;
					u_s = width * u_s;	// \in (-pw/2, pw/2)
					v_s = height * v_s;

					vec3f vd = view_dir;
					vec3f vu = cam_up;
					vec3f W, TxW, U, V;
					div_vec3f_by_scalar(&W, &vd, length_of_vec3f(&vd));
					cross_vec3f(&TxW, &vu, &W);
					div_vec3f_by_scalar(&U, &TxW, length_of_vec3f(&TxW));
					cross_vec3f(&V, &W, &U);

					vec3f dir = vec3f(0,0,0), tmp;
					mul_vec3f_by_scalar(&dir, &U, u_s);
					mul_vec3f_by_scalar(&tmp, &V, v_s);
					add_components_vec3f(&dir, &dir, &tmp);
					add_components_vec3f(&dir, &dir, &W);
					normalize_vec3f(&dir);

					vec3f pos_on_focal_plane = cam_pos + dir*(1.0f/(dir|view_dir))*focus_distance;
					float2 jitter = make_float2(0,0);
					int i=1;
					do {
						random = gi::next_random3f(uniform_random_01, (id+17*i)%(w*h));
						jitter = make_float2(random.z-0.5f, random.y-0.5f);
						if (i == 100) { jitter.x = jitter.y = 0; break; }
					} while (jitter.x*jitter.x + jitter.y*jitter.y > 1.0f);

					vec3f jitter_pos = cam_pos + U*jitter.x*aperture + V*jitter.y*aperture;
					dir = (pos_on_focal_plane - jitter_pos);
					normalize_vec3f(&dir);

					*this->direction(x, y) = dir;
					*this->origin(x, y) = jitter_pos;
				}
		}
		virtual std::string identification() { return "cuda version of the ray generator according to shirley."; }
		virtual void dont_forget_to_initialize_max_t() {}
	};

#endif

}



#endif

