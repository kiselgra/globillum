#pragma once
#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <iostream>
#include <fstream>


namespace rta{

inline float3 cosineSampleHemisphere(float u, float v){
                        float r  = sqrt(u);
                        float phi = 2.0f * M_PI * v;
                        float3 retVec;
                        retVec.x = r * cos(phi);
                        retVec.y = r * sin(phi);
                        retVec.z = sqrt(std::max(0.f,1.f-u));
                        return retVec;
                }
                inline float cos2sin(const float f) { return sqrt(std::max(0.f,1.f-f*f)); }
                inline float3 powerCosineSampleHemisphere(float u, float v, float exp){
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
      SPECULAR, DIFFUSE, TRANSMISSIVE, SPECULAR_REFLECTION, SPECULAR_TRANSMISSION, DISNEY_REFLECTION
                };

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
                                virtual void sample(const float3 &wo, float3 &wi, const float3 &sampleXYZ, float &pdfOut, bool enterGlass = false){
                                        wi.x = 0.0f;
                                        wi.y = 0.0f;
                                        wi.z = 0.0f;
                                        pdfOut = 1.f;//pdf(wo,wi,N);
                                }

                                //all input in world space.
                                virtual float pdf(const float3 &wo, const float3 &wi, const float3 &N) const {
                                        return 1.0f;
                                }
                                bool isDiffuse() const { return (_type == DIFFUSE);}
                                bool isSpecular() const {return (_type == SPECULAR);}
								bool isTransmissive()const {return (_type == SPECULAR_TRANSMISSION);}
								bool isReflective() const {return (_type == SPECULAR_REFLECTION || _type == DISNEY_REFLECTION);}
							//	bool isGlass() const{if(_type == SPECULAR_REFLECTION || _type == SPECULAR_TRANSMISSION) return true; return false ;}
                                protected:
                                        BRDF_TYPE _type;
                                        float3 _diffuse;
                                };
}

