#ifndef __GI_GPU_MATERIAL_H__ 
#define __GI_GPU_MATERIAL_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace rta {
	namespace cuda {

		struct texture_data {
			unsigned int w, h;
			unsigned char *rgba;
			int max_mm;
			enum where_t { host, device };
			where_t location;
			texture_data(int w, int h, where_t where) : w(w), h(h), rgba(0) {
				if (where == device) {
					checked_cuda(cudaMalloc(&rgba, 6*w*h));	// 6: 4 components times 1.5 for mip maps.
				}
				else
					rgba = new unsigned char[6*w*h];
				int min = w<h?w:h;
				max_mm = int(floor(log2f(float(min))));
				location = where;
			}
			texture_data() {
				w = h = 0; rgba = 0; max_mm = 0; location = host;
			}
			void upload(unsigned char *src) {
				checked_cuda(cudaMemcpy(rgba, src, 4*w*h, cudaMemcpyHostToDevice));
			}
			__host__ __device__ float3 sample_nearest(float s, float t) {
				float x = s*w;
				float y = (1.0f-t)*h;
				int nearest_x = int(x);
				int nearest_y = int(y);
				nearest_x = nearest_x % w;
				nearest_y = nearest_y % h;
				float3 a00 = make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+2])/255.0f);
				return a00;
			}
			__host__ __device__ float3 sample_bilin(float s, float t) {
				float x = s*w;
				float y = (1.0f-t)*h;
				int nearest_x = int(x);
				int nearest_y = int(y);
				int other_x = nearest_x+1;
				int other_y = nearest_y+1;
				if (x-floorf(x) < .5) other_x = nearest_x-1;
				if (y-floorf(y) < .5) other_y = nearest_y-1;
				// wx = 1.0 at floorf(x)=.5f.
				// wx = 0.5 at floorf(x)=.0f.
				float wx = fabsf(float(other_x)+.5 - x);
				float wy = fabsf(float(other_y)+.5 - y);
				nearest_x = nearest_x % w;
				nearest_y = nearest_y % h;
				other_x = other_x % w;
				other_y = other_y % h;
				float3 a00 = make_float3(float(rgba[4*(nearest_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+nearest_x)+2])/255.0f);
				float3 a01 = make_float3(float(rgba[4*(nearest_y*w+other_x)+0])/255.0f,
										 float(rgba[4*(nearest_y*w+other_x)+1])/255.0f,
										 float(rgba[4*(nearest_y*w+other_x)+2])/255.0f);
				float3 a10 = make_float3(float(rgba[4*(other_y*w+nearest_x)+0])/255.0f,
										 float(rgba[4*(other_y*w+nearest_x)+1])/255.0f,
										 float(rgba[4*(other_y*w+nearest_x)+2])/255.0f);
				float3 a11 = make_float3(float(rgba[4*(other_y*w+other_x)+0])/255.0f,
										 float(rgba[4*(other_y*w+other_x)+1])/255.0f,
										 float(rgba[4*(other_y*w+other_x)+2])/255.0f);
				float3 a0x = wx*a00 + (1.0f-wx)*a01;
				float3 a1x = wx*a10 + (1.0f-wx)*a11;
				return wy*a0x + (1.0f-wy)*a1x;
			}
			__host__ __device__ float3 sample_bilin_lod(float s, float t, float diff) {
				int W = w;
			   	int H = h;
				float lod = log2f(diff * fmaxf(w,h));
				lod = fminf(lod, float(max_mm));
				unsigned int offset = 0;
				for (int i = 0; i < lod; ++i) {
					offset += W*H;
					W = (W+1)/2;
					H = (H+1)/2;
				}
				if (s < 0)  s += -truncf(s)+1;
				if (s >= 1) s -= truncf(s);
				if (t < 0)  t += -truncf(t)+1;
				if (t >= 1) t -= truncf(t);
				float x = s*W;
				float y = (1.0f-t)*H;
				int nearest_x = int(x);
				int nearest_y = int(y);
				int other_x = nearest_x+1;
				int other_y = nearest_y+1;
				if (x-floorf(x) < .5) other_x = nearest_x-1;
				if (y-floorf(y) < .5) other_y = nearest_y-1;
				// wx = 1.0 at floorf(x)=.5f.
				// wx = 0.5 at floorf(x)=.0f.
				float wx = fabsf(float(other_x)+.5 - x);
				float wy = fabsf(float(other_y)+.5 - y);
				// nearest is in bounds.
// 				printf("a) x=%04d y=%04d ox=%04d oy=%04d\n", nearest_x, nearest_y, other_x, other_y);
				if (other_x > W) other_x -= W;
				if (other_x < 0) other_x += W;
				if (other_y > H) other_y -= H;
				if (other_y < 0) other_y += H;
// 				printf("b) x=%04d y=%04d ox=%04d oy=%04d\n", nearest_x, nearest_y, other_x, other_y);
				float3 a00 = make_float3(float(rgba[4*(offset + nearest_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+nearest_x)+2])/255.0f);
				float3 a01 = make_float3(float(rgba[4*(offset + nearest_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset + nearest_y*W+other_x)+2])/255.0f);
				float3 a10 = make_float3(float(rgba[4*(offset + other_y*W+nearest_x)+0])/255.0f,
										 float(rgba[4*(offset + other_y*W+nearest_x)+1])/255.0f,
										 float(rgba[4*(offset + other_y*W+nearest_x)+2])/255.0f);
				float3 a11 = make_float3(float(rgba[4*(offset + other_y*W+other_x)+0])/255.0f,
										 float(rgba[4*(offset + other_y*W+other_x)+1])/255.0f,
										 float(rgba[4*(offset + other_y*W+other_x)+2])/255.0f);
				float3 a0x = wx*a00 + (1.0f-wx)*a01;
				float3 a1x = wx*a10 + (1.0f-wx)*a11;
				return wy*a0x + (1.0f-wy)*a1x;
			}
		};
			
			void compute_mipmaps(texture_data *tex);

			struct material_t {
				float3 diffuse_color, specular_color;
				float alpha;
				texture_data *diffuse_texture;
				texture_data *specular_texture;
				texture_data *alpha_texture;

				material_t() 
				: diffuse_color(make_float3(0,0,0)), specular_color(make_float3(0,0,0)), alpha(1), diffuse_texture(0), specular_texture(0), alpha_texture(0) {
				}
				float3 diffuseColor(const float2 &TC, const float2 &upper_T, const float2 &right_T)const{
					float3 diffuse = diffuse_color;
					if (diffuse_texture ) {
                                        float diff_x = fabsf(TC.x - upper_T.x);
                                        float diff_y = fabsf(TC.y - upper_T.y);
                                        diff_x = fmaxf(fabsf(TC.x - right_T.x), diff_x);
                                        diff_y = fmaxf(fabsf(TC.y - right_T.y), diff_y);
                                        float diff = fmaxf(diff_x, diff_y);
                                        diffuse *= diffuse_texture->sample_bilin_lod(TC.x, TC.y, diff);
                                        }
					return diffuse;
				}

				float3 specularColor(const float2 &TC, const float2 &upper_T, const float2 &right_T)const{
                                        float3 specular  = specular_color;
                                        if (specular_texture ) {
                                        float diff_x = fabsf(TC.x - upper_T.x);
                                        float diff_y = fabsf(TC.y - upper_T.y);
                                        diff_x = fmaxf(fabsf(TC.x - right_T.x), diff_x);
                                        diff_y = fmaxf(fabsf(TC.y - right_T.y), diff_y);
                                        float diff = fmaxf(diff_x, diff_y);
                                        specular  *= specular_texture->sample_bilin_lod(TC.x, TC.y, diff);
                                        }
					return specular;
                                }

			};

			//! convert standard rta cpu tracer materials to a version suitable for gpu use.
			material_t* convert_and_upload_materials(int &N);
			//! we do it this way because we compute mipmaps on the gpu.
			cuda::material_t* download_materials(cuda::material_t *gpumat, int N);
			
			
			//! evaluate material using point sampling with bilinear interpolation.
			void evaluate_material_bilin(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
										 cuda::material_t *mats, float3 *dst, float3 background);

			/*! \brief evaluate material using ray differentials computed via neighboring rays.
			 *  \attention this only works for camera rays!
			 */
			void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
								   cuda::material_t *mats, float3 *dst, float *ray_org, float *ray_dirs, float3 background);
			/*! \brief evaluate material using ray differentials stored in rta container (ray_diff_org,_dir are expected to be of size w * (2h).
			 */
			void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
								   cuda::material_t *mats, float3 *dst, float *ray_org, float *ray_dir, float *ray_diff_org, float *ray_diff_dir, 
								   float3 background);
		}
		
		//! cpu material evaluation. all pointers must be downloaded, already.
		void evaluate_material(int w, int h, triangle_intersection<cuda::simple_triangle> *ti, cuda::simple_triangle *triangles, 
							   cuda::material_t *mats, float3 *dst, float3 *ray_org, float3 *ray_dir, 
						   float3 *ray_diff_org, float3 *ray_diff_dir, float3 background);


		inline float3 cosineSampleHemisphere(float u, float v, const float3 &N){
			float cosTheta  = sqrt(u);
			float phi = 2.0f * M_PI * v;
			float sinTheta = sqrt(1.0f-u);
			float3 retVec;
			retVec.x = sinTheta * cos(phi);
			retVec.y = sinTheta * sin(phi);
			retVec.z = cosTheta;
			return retVec;
		}
		inline float cos2sin(const float f) { return sqrt(std::max(0.f,1.f-f*f)); }
		inline float3 powerCosineSampleHemisphere(float u, float v, const float3 &N, float exp){
			float phi = float(2.0f * M_PI) * u;
			float cosTheta = powf(v,1.0f/(exp+1.f));
			float sinTheta = cos2sin(cosTheta);
			float3 retVec;
			retVec.x = cos(phi) * sinTheta;
			retVec.y = sin(phi) * sinTheta;
			retVec.z = cosTheta;
			return retVec;
		}		
		inline float3 reflectR (const float3 &v, const float3 &n){
  			return v - 2.f * (v|n) * n;
		}
			
		inline float clamp01(float a){ return (a<0.f? 0.0f : a );}//(a>1.f? 1.0f : a));}	
		enum BRDF_TYPE{
			SPECULAR, DIFFUSE, TRANSMISSIVE
		};
//		namespace gi{
			class Material{
				public:
				Material(){}
				// all input in world space.
				virtual float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const{					
					return float3();
				}
				//sample direction based on material's brdf
				// returns :
				// wi : sampled direction in Tangent space
				// pdfOut : corresponding pdf to sampled direction
				// float3 : brdf value (no return value because we cannot switch to tangent space)
				virtual void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, const float3 &N, float &pdfOut){
					wi.x = 0.0f;
					wi.y = 0.0f;
					wi.z = 0.0f;
					pdfOut = pdf(wo,wi,N);
				}
				
				//all input in world space.
				virtual float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
					return 1.0f;
                		}
				bool isDiffuse() const { return (_type == DIFFUSE);}
				bool isSpecular() const {return (_type == SPECULAR);}
				protected:
					BRDF_TYPE _type;
					float3 _diffuse;
				};

			class BlinnMaterial : public Material {
			public:
				BlinnMaterial(const rta::cuda::material_t* mat, const float2 &T, const float2 &upperT, const float2 &rightT):Material(),_mat(mat),_shininess(40.0f){
					_specular = _mat->specularColor(T,upperT,rightT);
					_diffuse = _mat->diffuseColor(T,upperT,rightT);
					float sumDS = _diffuse.x + _diffuse.y + _diffuse.z + _specular.x + _specular.y + _specular.z;
					if(sumDS > 1.f){
						_diffuse /= sumDS;
						_specular /= sumDS;
					}
					_type = DIFFUSE;
				}

			  float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const{
				if(_type == SPECULAR){  
				//specular case
					float3 R = reflectR(wi,N);
			 		float3 brdf = _diffuse * float(M_1_PI) + (_shininess + 1)*_specular * 0.5 * M_1_PI * pow(clamp01(R|wo), _shininess);
                        	}else{
				//diffuse case
					return _diffuse * (1.0f/M_PI) * clamp01(wi|N);
				}        
			}
                        void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, const float3 &N, float &pdfOut) {
				float pd = _diffuse.x + _diffuse.y + _diffuse.z;
				float ps = _specular.x + _specular.y + _specular.z;
				float sumPds = pd + ps;
				if (sumPds < 1.f){
					pd /= sumPds;
					ps /= sumPds;
				}
				if(sampleXYZ.z < pd){
 					_type = DIFFUSE;
                                        wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y,N);
                                        pdfOut = pd * clamp01(wi.z) * (1.0f/M_PI);
				}
				else { 
					_type = SPECULAR;
					wi = powerCosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y, N, _shininess);
					float dotWN = wi.z;
					pdfOut =  ( dotWN < 0.0f ? 0.0f : ps*(_shininess+1.0f)*powf(dotWN,_shininess)*float(1.0f/(2.0f*M_PI)) );
				}
	                    }
                        float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
                        	if(_type == SPECULAR){
					float dotWN = (wi|N);
					return ( dotWN < 0.0f ? 0.0f : (_shininess+1.0f)*powf(dotWN,_shininess)*float(1.0f/(2.0f*M_PI)) );
				}
				return clamp01(wi|N)*(1.0f/M_PI);      
			}
			protected:
			const rta::cuda::material_t* _mat;
			float3 _specular;
			float _shininess;
             };
	

		

		class LambertianMaterial : public Material{
			public:
			LambertianMaterial(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT):Material(),_mat(mat){
                                _type = DIFFUSE;
				_diffuse = _mat->diffuseColor(T,upperT,rightT);
                                float sumDS = _diffuse.x + _diffuse.y + _diffuse.z;
                                if(sumDS > 1.f){
                                	_diffuse /= sumDS;
                     		}
			}
			// evaluates brdf based on in/out directions wi/wo in world space
			float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const{
                       		return _diffuse * (1.0f/M_PI) * clamp01(wi|N);
                        }
			//returns sampled direction wi in Tangent  space.
                        void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, const float3 &N, float &pdfOut) { 
				wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y,N);
                             	pdfOut = clamp01(wi.z) * (1.0f/M_PI);
                        }
			//computes pdf based on wi/wo in world space
                        float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
				return clamp01(wi|N) * (1.0f/M_PI);               
                        }
			private:
			const rta::cuda::material_t *_mat;
		};

/*		struct PrincipledParameters{
		float metallic ;
		float subsurface;
		float specular ;
		float roughness;
		float specularTint;
		float anisotropic;
		float sheen;
		float sheenTint;
		float clearcoat;
		float clearcoatGloss;
		float opacity;
		float transmiss;
		float etaInside;
		float3 color;
		float3 tangent;
		};
		class PrincipledMaterial : public Material{

		}
	*/		
}

#endif

