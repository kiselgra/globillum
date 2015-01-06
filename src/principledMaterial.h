#pragma once
#include "principledParameters.h"
#include "materialBase.h"
namespace rta{

namespace Principled{
struct Sample3f{
        Sample3f(){}
        Sample3f(const float3 &a, float b){
                value = a;
                pdf = b;
        }
	Sample3f& operator=(const Sample3f &other){value = other.value; pdf = other.pdf; return *this;}
        float3 value;
        float pdf;
};

 inline float dot(const float3 &a, const float3 &b) { return (a|b);}
inline float saturate(float a) {
	  return clamp(a,0.0f,1.0f);
  }
  // Used by Disney BRDF.
  inline float SchlickFresnel(float u) 
  {
	  float m = saturate( 1.0f - u);
	  float m2 = m*m;
	  return m2*m2*m;
  }
 inline float sqr(float a) {return (a*a);}
inline float3 normalize (const float3 &a){
 float b = 1.0f/sqrt(dot(a,a));
 return (a*b);
}
  //from "Disney Principled BRDF Notes": Sampling of GTR2 Aniso Specular Lobe (formula 13)
  inline float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
  {
	  return 1 / ( float(M_PI) * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
  }


  //from "Disney Principled BRDF Notes": Sampling of GTR2 Aniso Specular Lobe (formula 14+15)
  inline Sample3f sampleGTR2AnisoHalfwayVector(float u, float v, float alphaX2, float alphaY2, const float3 &N,const float3 &T, const float3 &Bi){
	  float sinPiEta = sinf(2.0f*M_PI * u); 
	  float cosPiEta = cosf(2.0f*M_PI * u);

	  float fac = sqrt(v/(1.0f-v));
	  float3 h = normalize(fac * (alphaX2*cosPiEta*T + alphaY2*sinPiEta*Bi) + N);
	  
	  float HdotN = (h|N);
	  float HdotT = (h|T);
	  float HdotB = (h|Bi);
	  
	  float w = GTR2_aniso(HdotN,HdotT, HdotB, alphaX2, alphaY2);
	  
	  return (Sample3f(h, w));
  }


  inline float smithG_GGX(float Ndotv, float alphaG) 
  {
	  float a = alphaG*alphaG;
	  float b = Ndotv*Ndotv;
	  return 1.f/(Ndotv + sqrt(a + b - a*b));
  }
  
  // formula for Xi  in paper M.Models for Refraction (B.Walter et.al)
  inline float Xi(float s){
	  return (s>0.0f? 1.0f : 0.0f);
  }
  // formula for Geometric Term  in paper M.Models for Refraction (B.Walter et.al)
  inline float G1(const float3 &v, const float3 &m, const float3 &N,float alphaSquare, float tanthetaV){ // thetaV = angle(view, normal)
	  float fac = Xi((v|m)/(v|N));
	  return fac * 2.0f/(1.0f + sqrt(1.0f + alphaSquare * tanthetaV*tanthetaV));
  }
  
  inline float signCalc(float a){
	  return (a>0? 1.0f : -1.0f);
	  }

  // sampling of GTR2Aniso specular Lobe
  inline Sample3f sampleGTR2Aniso(const float& u, const float& v, const float& alphaX2, const float& alphaY2, const float3 &Nshading,  const float3 &T, 
	  const float3 &Bi, const float3 &V,float metallic, BRDF_TYPE &brdfType, float ior) {

	  // sample specular highlight reflection;
	  Sample3f h = sampleGTR2AnisoHalfwayVector(u,v, alphaX2,alphaY2, Nshading,T,Bi);

	  //rename variables for consistency with M.Models for Refraction (B.Walter et.al)
	  const float3 &m =  h.value ;
	  const float3 &n =  Nshading ;
	  float c = dot(m,V);

	  float3 L ; //outgoing light vector.
	  float fresnel = SchlickFresnel(c);
	  if(ior == 0 || u < fresnel){
		  //reflect
		  brdfType = SPECULAR_REFLECTION;
		  L = reflectR(V,m);
	  }else{
		  float totalInternal = 1 + ior*(c*c - 1);
		  if(totalInternal <= 0.0f && ior > 1.0f){
			  //total internal reflection.
			float3 val = make_float3(0.0f,0.0f,0.0f);
			  return Sample3f(val, 0.0f);
		  }
		  //refract
		  brdfType = SPECULAR_TRANSMISSION;
		  L = (ior*c - signCalc(dot(V,n))*sqrt(totalInternal))*m - ior*V;
	  }
	  

	  // map roughness (Disney notes)
	  float roughg = sqr(alphaX2*alphaY2*.5+.5);
	  ////// precomputations for formular 34 from Microfacet Models for Refraction (B.Walter)
	  float dotLm = dot(L,m);
	  float dotVm = dot(V,m);
	  
	  float tanThetaL = sqrt(1-dotLm*dotLm)/dotLm;
	  float tanThetaV = sqrt(1-dotVm*dotVm)/dotVm;
	  ////// formula 23 from Microfacet Models for Refraction (B.Walter)
	  //// G1 estimate is formula 34
	  float Gs = G1(L,m,n,roughg*roughg,tanThetaL) * G1(V,m,n,roughg*roughg,tanThetaV);

	  ////// formula 41 from paper Microfacet Models for Refraction (B.Walter et.al)
	  //// weight computation is the same for reflection and refraction.
	  float lweight = (abs(dot(L,m))*Gs)/(abs(dot(L,n)) * abs(dot(n,m))) ;
	  return Sample3f(L,lweight);
  }

  inline Sample3f samplePrincipledSpecular(const float& u, const float& v, const PrincipledBRDFParameters& brdfParams, const float3 &N, const float3 &wo,BRDF_TYPE &brdfType, float ior){
  		float aspect = sqrt(1-brdfParams.anisotropic*.9);
  		float ax = std::max(.001f, sqr(brdfParams.roughness)/aspect);
  		float ay = std::max(.001f, sqr(brdfParams.roughness)*aspect);
		float3 Tx,Ty; // TODO : Correct TX,TY!
  		return sampleGTR2Aniso(u,v, ax, ay, N,Tx, Ty, wo, brdfParams.metallic,brdfType,ior); 
	}
}
class PrincipledMaterial : public Material{
                        public:
                        PrincipledMaterial(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT):Material(),_mat(mat){
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
                        //        wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y,N);
                        //        pdfOut = clamp01(wi.z) * (1.0f/M_PI);
				float _ior = 1.0f;
				Principled::Sample3f sam = Principled::samplePrincipledSpecular(sampleXYZ.x, sampleXYZ.y, (*_mat->parameters), N, wo, _type, _ior);
				wi = sam.value;
                        }
                        //computes pdf based on wi/wo in world space
                        float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
                                return clamp01(wi|N) * (1.0f/M_PI);
                        }
                        private:
                        const rta::cuda::material_t *_mat;
                };
}
