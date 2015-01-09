#pragma once

#include "principledParameters.h"
#include "materialBase.h"

#include <libhyb/trav-util.h>

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
  inline Sample3f sampleGTR2AnisoHalfwayVector(float u, float v, float alphaX2, float alphaY2,const float3 &T, const float3 &Bi){
	  float sinPiEta = sinf(2.0f*M_PI * u); 
	  float cosPiEta = cosf(2.0f*M_PI * u);
	float3 N = make_float3(0.f,0.f,1.f); // 
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
//	if(fac == 0.0f) std::cerr<<"Xi from "<<(v|m)<<" and "<<(v|N)<<"\n";
	  return fac * 2.0f/(1.0f + sqrt(1.0f + alphaSquare * tanthetaV*tanthetaV));
  }
  
  inline float signCalc(float a){
	  return (a>0? 1.0f : -1.0f);
	  }

  // sampling of GTR2Aniso specular Lobe
  inline Sample3f sampleGTR2Aniso(const float& u, const float& v, const float& alphaX2, const float& alphaY2, const float3 &T, 
	  const float3 &Bi, const float3 &V,float metallic, BRDF_TYPE &brdfType, float ior) {

	  // sample specular highlight reflection;
	  Sample3f h = sampleGTR2AnisoHalfwayVector(u,v, alphaX2,alphaY2, T,Bi);

	  //rename variables for consistency with M.Models for Refraction (B.Walter et.al)
	  const float3 &m =  h.value ;
	  const float3 &n =  make_float3(0.f,0.f,1.f);//Nshading ;
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
	  //flip if neccessary
//	  float3 VV = V;
	  if(dot(L,n) < 0){
		 L = -1.0f*L;
		//VV = -1.0f * VV;
	}	
	  // map roughness (Disney notes)
	  float roughg = sqr(alphaX2*alphaY2*.5+.5);
	  ////// precomputations for formular 34 from Microfacet Models for Refraction (B.Walter)
	  float dotLm = dot(L,m);
	  float dotVm = dot(V,m);
	  if(dotLm == 0 || dotVm == 0) 
		std::cerr<<"Warning: Division by 0 : "<<dotLm << " or " << dotVm << "\n"; 
	  float tanThetaL = sqrt(1-dotLm*dotLm)/dotLm;
	  float tanThetaV = sqrt(1-dotVm*dotVm)/dotVm;
	  ////// formula 23 from Microfacet Models for Refraction (B.Walter)
	  //// G1 estimate is formula 34
	  float Gs = G1(L,m,n,roughg*roughg,tanThetaL) * G1(V,m,n,roughg*roughg,tanThetaV);

	  ////// formula 41 from paper Microfacet Models for Refraction (B.Walter et.al)
	  //// weight computation is the same for reflection and refraction.
	  float lweight = (fabs(dot(L,m))*Gs)/(fabs(dot(L,n)) * fabs(dot(n,m))) ;
//if( lweight == 0 ) 
//		std::cerr<<"Lweight "<<lweight<<" from "<<dot(L,n)<<" , "<<dot(n,m)<<" and "<<dot(L,m)<<" and "<<dot(V,m)<<" and "<<Gs<<" and div "<< (fabs(dot(L,m))*Gs) <<" / "<< (fabs(dot(L,n)) * fabs(dot(n,m)))<<" iand and " << dot(n,m) <<"\n";
//		std::cerr<<"Lweight is 0 Warning: Division by 0 : "<<lweight<< "\n"; 
	  return Sample3f(L,lweight);
  }

  inline Sample3f samplePrincipledSpecular(const float& u, const float& v, const PrincipledBRDFParameters& brdfParams, const float3 &wo,BRDF_TYPE &brdfType, float ior, const float3 &Tx, const float3 &Ty){
  		float aspect = sqrt(1-brdfParams.anisotropic*.9);
  		float ax = std::max(.001f, sqr(brdfParams.roughness)/aspect);
  		float ay = std::max(.001f, sqr(brdfParams.roughness)*aspect);
		//float3 Tx,Ty; // TODO : Correct TX,TY!
  		return sampleGTR2Aniso(u,v, ax, ay, Tx, Ty, wo, brdfParams.metallic,brdfType,ior); 
	}

	inline float GTR1(float NdotH, float a)
	{
		if (a >= 1) return 1/float(M_PI);
		float a2 = a*a;
		float t = 1 + (a2-1)*NdotH*NdotH;
		return (a2-1) / (float(M_PI)*log(a2)*t);
	}


	inline float3 mix(const float3 &x, const float3 &y, float s) {
		return (1.0f-s)*x + s*y;
	}

	inline float mix(float x, float y, float s) {
		return (1.0f-s)*x + s*y;
	}


	inline float3 mon2lin(const float3 &p) 
	{
		float3 a = make_float3(pow(p.x, 2.2f), pow(p.y, 2.2f), pow(p.z, 2.2f));
		return a;
	}

	float3 evaluatePrincipledBRDF_specular(const float3 &wo, const float3 &N, const float3 &wi, const float3 &R,const PrincipledBRDFParameters& brdfParams){
		//float3 &N = dg.Ns;
		const float3 V = wo;
		const float3 L = wi;

		float NdotL = dot(N,L);
		float NdotV = dot(N,V);
		if (NdotL < 0 || NdotV < 0){
			std::cerr<<"Evaluate negative "<<NdotL<<" and "<<NdotV<<"\n";
			return make_float3(0.0f,0.0f,0.0f);
		}
		
		float3 H = normalize(L+V);
		float NdotH = dot(N,H);
		float LdotH = dot(L,H);

		float3 baseColor = make_float3(R.x,R.y,R.z);
		float3 Cdlin = mon2lin(baseColor);
		float Cdlum = .3*Cdlin.x + .6*Cdlin.y + .1*Cdlin.z; // luminance approx.

		float3 whiteColor = make_float3(1.f,1.f,1.f);
		float3 Ctint = Cdlum > 0 ? Cdlin/Cdlum : whiteColor; // normalize lum. to isolate hue+sat
		float3 Cspec0 = mix(brdfParams.specular*.08*mix(whiteColor, Ctint, brdfParams.specularTint), Cdlin, brdfParams.metallic);

		float FH = SchlickFresnel(LdotH);

		//// specular
		float3 Fs = mix(Cspec0, whiteColor, FH);

		// clearcoat (ior = 1.5 -> F0 = 0.04)
		float cg = 0.0f;
		if(brdfParams.clearcoat > 0.0f){
			float Dr = GTR1(NdotH, mix(.1,.001,brdfParams.clearcoatGloss));
			float Fr = mix(0.04f, 1.0f, FH);
			float Gr = smithG_GGX(NdotL, 0.25f) * smithG_GGX(NdotV, 0.25f);
			cg = 0.25f*brdfParams.clearcoat*Gr*Fr*Dr;
		}

		float3 specColor = Fs;// Gs * Fs *Ds;
		float3 clearcoatColor = make_float3(cg,cg,cg);

		float3 sumColor = (specColor + clearcoatColor);
		return make_float3(sumColor.x,sumColor.y,sumColor.z)*(NdotL)*brdfParams.metallic;
	}

	float3 evaluatePrincipledBRDF_diffuse(const float3& wo, const float3 &N, const float3& wi, const float3& R,const PrincipledBRDFParameters& brdfParams){
		
		const float3& V = wo;
		const float3& L = wi;

		float NdotL = dot(N,L);
		float NdotV = dot(N,V);
		if (NdotL < 0 || NdotV < 0){
			return make_float3(0.0f,0.0f,0.0f);
		}

		float3 H = normalize(L+V);
		float NdotH = dot(N,H);
		float LdotH = dot(L,H);

		float3 baseColor = make_float3(R.x,R.y,R.z);
		float3 Cdlin = mon2lin(baseColor);
		float Cdlum = .3*Cdlin.x + .6*Cdlin.y + .1*Cdlin.z; // luminance approx.

		float3 whiteColor = make_float3(1.f,1.f,1.f);
		float3 Ctint = Cdlum > 0 ? Cdlin/Cdlum : whiteColor; // normalize lum. to isolate hue+sat

		float3 Csheen = mix(whiteColor, Ctint, brdfParams.sheenTint);

		// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
		// and mix in diffuse retro-reflection based on roughness
		float FL = SchlickFresnel(NdotL);
		float FV = SchlickFresnel(NdotV);
		float Fd90 = 0.5f + 2.0f * LdotH*LdotH * brdfParams.roughness;
		float Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV);

		float FH = SchlickFresnel(LdotH);
		//// sheen
		float3 Fsheen = FH * brdfParams.sheen * Csheen;
		//
		//// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
		//// 1.25 scale is used to (roughly) preserve albedo
		//// Fss90 used to "flatten" retroreflection based on roughness
		float Fss90 = LdotH*LdotH*brdfParams.roughness;
		float Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV);
		float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

		float3 diffuseColor =  ((1.0f/float(M_PI)) * mix(Fd, ss, brdfParams.subsurface)*Cdlin + Fsheen) * (1.0f-brdfParams.metallic);

		float3 sumColor = diffuseColor;

		return make_float3(sumColor.x,sumColor.y,sumColor.z)*(NdotL);
	}

}
class PrincipledMaterial : public Material{
                        public:

						PrincipledMaterial():Material(),_mat(0){_type = DIFFUSE; _diffuse = make_float3(0.f,0.f,0.f); _Tx = make_float3(0.f,0.f,0.f); _Ty = make_float3(0.f,0.f,0.f);}
                        PrincipledMaterial(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT, const float3 &Tx, const float3 &Ty):Material(),_mat(mat){
                                _type = DIFFUSE;
                                _diffuse = _mat->diffuseColor(T,upperT,rightT);
				_Tx = Tx;
				_Ty = Ty;
                        }

				void init(const rta::cuda::material_t *mat, const float2 &T, const float2 &upperT, const float2 &rightT, const float3 &Tx, const float3 &Ty){
					_mat = mat;
					_type = DIFFUSE;
					_diffuse = _mat->diffuse_color;//_mat->diffuseColor(T,upperT,rightT);
					_Tx = Tx;
					_Ty = Ty;
				}
                        // evaluates brdf based on in/out directions wi/wo in world space
                        float3 evaluate(const float3 &wo, const float3 &wi, const float3& N) const{
				if(_type == SPECULAR_REFLECTION || _type == SPECULAR_TRANSMISSION){
					return Principled::evaluatePrincipledBRDF_specular(wo, N, wi, _diffuse, (*_mat->parameters));				
				}
				return Principled::evaluatePrincipledBRDF_diffuse(wo,N,wi,_diffuse,(*_mat->parameters));
                        }
                        //returns sampled direction wi in Tangent  space.
                        void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, float &pdfOut) {
				if(sampleXYZ.z < _mat->parameters->metallic){
					// do specular reflection to get metallic look
					_type = SPECULAR;
					float _ior = 0.f;
					float3 Tx_ts = make_float3(1.f,0.f,0.f);
					float3 Ty_ts = make_float3(0.f,1.f,0.f);
					Principled::Sample3f sam = Principled::samplePrincipledSpecular(sampleXYZ.x, sampleXYZ.y, (*_mat->parameters), wo, _type, _ior, Tx_ts, Ty_ts);
					pdfOut = sam.pdf ;//* _mat->parameters->metallic;
					wi = sam.value;
					if(pdfOut == 0) {
						wi = cosineSampleHemisphere(sampleXYZ.x, sampleXYZ.y);
						_type = DIFFUSE;
						pdfOut = clamp01(wi.z) / M_PI;
                                        }

				}else{
					//do diffuse sampling
					_type = DIFFUSE;
					wi = cosineSampleHemisphere(sampleXYZ.x,sampleXYZ.y);
					pdfOut = clamp01(wi.z) * (1.0f/M_PI);	
					if(pdfOut == 0) {
						std::cerr<<"Warning: pdf is 0 causes division by 0.\n";
					}
				}
                        }
                        //computes pdf based on wi/wo in world space
                        float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
                                return clamp01(wi|N) * (1.0f/M_PI);
                        }
                        private:
                        const rta::cuda::material_t *_mat;
			float3 _Tx;
			float3 _Ty;
                };
}
