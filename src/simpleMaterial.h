#pragma once
#include "materialBase.h"

namespace rta {

	class BlinnMaterial : public Material {
	public:
		BlinnMaterial():Material(),_mat(0), _shininess(40.f){
			_specular = make_float3(0.f,0.f,0.f);
			_diffuse = make_float3(0.f,0.f,0.f);
			_type = DIFFUSE;
		}
		BlinnMaterial(const rta::cuda::material_t* mat, const float2 &T, const float2 &upperT, const float2 &rightT)
		: Material(),_mat(mat),_shininess(40.0f) {
			_specular = _mat->specularColor(T,upperT,rightT);
			_diffuse = _mat->diffuseColor(T,upperT,rightT);
			float sumDS = _diffuse.x + _diffuse.y + _diffuse.z + _specular.x + _specular.y + _specular.z;
			if(sumDS > 1.f){
				_diffuse /= sumDS;
				_specular /= sumDS;
			}
			_type = DIFFUSE;
		}
		void init (const rta::cuda::material_t* mat, const float2 &T, const float2 &upperT, const float2 &rightT) {
			_mat = mat;
			_shininess = 40.f;
			_specular = _mat->specularColor(T,upperT,rightT);
			_diffuse = _mat->diffuseColor(T,upperT,rightT);
			float sumDS = _diffuse.x + _diffuse.y + _diffuse.z + _specular.x + _specular.y + _specular.z;
			if(sumDS > 1.f){
				_diffuse /= sumDS;
				_specular /= sumDS;
			}
			_type = DIFFUSE;
		}

		float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const {
			if(_type == SPECULAR){
				//specular case
				float3 R = reflectR(wi,N);
				float3 brdf = _diffuse * float(M_1_PI) + (_shininess + 1)*_specular * 0.5 * M_1_PI * pow(clamp01(R|wo), _shininess);
			}else{
				//diffuse case
				return _diffuse * (1.0f/M_PI) * clamp01(wi|N);
			}
		}
		void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, float &pdfOut, bool enterGlass) {
			float pd = _diffuse.x + _diffuse.y + _diffuse.z;
			float ps = _specular.x + _specular.y + _specular.z;
			float sumPds = pd + ps;
			if (sumPds < 1.f){
				pd /= sumPds;
				ps /= sumPds;
			}
			if(sampleXYZ.z < pd){
				_type = DIFFUSE;
				wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y);
				pdfOut = pd * clamp01(wi.z) * (1.0f/M_PI);
			}
			else {
				_type = SPECULAR;
				wi = powerCosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y, _shininess);
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



	class LambertianMaterial : public Material {
	public:
		LambertianMaterial():Material(),_mat(0){ _type = DIFFUSE; _diffuse = make_float3(0.f,0.f,0.f);}
		LambertianMaterial(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT)
		: Material(),_mat(mat) {
			_type = DIFFUSE;
			_diffuse = _mat->diffuseColor(T,upperT,rightT);
		}

		void init(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT) {
			_mat = mat;
			_type = DIFFUSE;
			_diffuse = _mat->diffuseColor(T,upperT,rightT);//*mat->parameters->color;
		}

		// evaluates brdf based on in/out directions wi/wo in world space
		float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const {
			return _diffuse * (1.0f/M_PI) * clamp01(wi|N);
		}
		//returns sampled direction wi in Tangent  space.
		void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, float &pdfOut, bool enterGlass) {
			wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y);
			pdfOut = clamp01(wi.z) * (1.0f/M_PI);
		}
		//computes pdf based on wi/wo in world space
		float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
			return clamp01(wi|N) * (1.0f/M_PI);
		}
	private:
		const rta::cuda::material_t *_mat;
	};
}
