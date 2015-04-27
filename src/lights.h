#ifndef __GI_LIGHTS_H__ 
#define __GI_LIGHTS_H__ 

#include <librta/cuda-kernels.h>
#include <librta/cuda-vec.h>

#include <vector>
#include <algorithm>

namespace gi {

	inline int clamp(int a, int lower, int upper){
		int ret = (a>lower)? a : lower;
		return (ret > upper) ? upper : ret;
	}
	// from pbrt: Monte Carlo Rendering with Natural Illumination
	// http://web.cs.wpi.edu/~emmanuel/courses/cs563/S07/projects/envsample.pdf
	class Distribution1D{
	public:
		Distribution1D() : PDF(0), CDF(0), size(0) {}
		~Distribution1D(){
			delete[] PDF; PDF = 0;
			delete[] CDF; CDF = 0;
		}
		float init(const float* data, const int size){
			this->size = size;
			PDF = new float[size];
			CDF = new float[size+1];
			// sum up all input data.
			float sum = 0.0f;
			for(int i = 0; i < size; i++){
				sum += data[i];
			}
			//avoid division by zero
			float scale = 0.0f;
			if(sum > 0.0f) scale = 1.0f/sum;
			CDF[0] = 0.0f;
			for(int i=0; i<size; i++){
				PDF[i] = data[i] * scale;
				if(i > 0) CDF[i] = (CDF[i-1] + PDF[i-1]);
				
			}
			CDF[size] = 1.0f;
			return sum;
		}
		//find index such that CDF[index]<= randU < CDF[index+1]
		int binarySearchCDF(float randU) {	
			float* b = std::upper_bound(CDF, CDF + size, randU);
			int index = clamp(int(b-CDF)-1,0,size-1);
			return index;
		}
		void sample(const float randU, float &x, float &p) {
			int idx = binarySearchCDF(randU);	
			float t = (CDF[idx+1]-randU)/(CDF[idx+1] - CDF[idx]);
			x = (1.0-t) * idx + t*(idx+1);
			p = PDF[idx];
		}
	protected:
		int size;
		float* PDF;
		float* CDF;
	};

	class Distribution2D{
	public:
		Distribution2D(const float* data, const int w, const int h) {
			xDistr = new Distribution1D[h];
			width = w;
			height = h;
			std::vector<float> colSum(h);
			for(int y=0; y < h; y++){
				float sum = xDistr[y].init(&data[y*w],w);
				colSum[y] = sum;
			}
			yDistr.init(&colSum[0],h);
		}
		~Distribution2D(){
			delete[] xDistr; xDistr = 0;
		}
		void sample(const float randU, const float randV, float &outU, float &outV, float &outPdf) {
			float yPdf, xPdf;
			yDistr.sample(randU, outU, yPdf);
			int idx = clamp(int(outU),0,height-1);
			xDistr[idx].sample(randV,outV,xPdf);
			outPdf = yPdf * xPdf;
		}
	protected:
		int width;
		int height;
		Distribution1D yDistr;			//distribution to select row.
		Distribution1D* xDistr;	//distribution to select value from a row distribution.
	};

	struct rect_light {
		float3 center, dir, up, col;
		float2 wh;
	};

	struct sky_light {
		float scale;
		char *map;
		float3 *data;
		int w, h;
		/*! we use a ptr here because we don't want to spead the union's (\ref gi::light) constraints (trivially constructable)
		 *  to the distribution's code. */
		Distribution2D *lightDistribution;
		void initSkylight(){
			float* importance = new float[w*h];
			for(int y=0; y<h; y++){
				for(int x=0; x<w; x++){
					//importance: sum of light RGB * importance compensation for poles 
					importance[y*w+x] = (sinf(M_PI * (y+0.5f)/h)) * ((data[y*w+x].x + data[y*w+x].y + data[y*w+x].z));
				}

			}
			lightDistribution = new Distribution2D(importance,w,h);
			delete [] importance;
		}
		void deleteSkylight() {
			delete lightDistribution;
		}
		float3 toDirection(float theta, float phi) const{
			float3 a;
			a.x = sinf(theta) * cosf(phi);
			a.y = sinf(theta) * sinf(phi);
			a.z = cosf(theta);	
			return a;
		}
		float3 sample(const float randU, const float randV, float &outPdf, float3 &outDir){
			float u,v,pdf;
			lightDistribution->sample(randU, randV, u, v, pdf);
			float theta = u*M_PI/float(h);
			float phi = v*2.0f*M_PI/float(w);
			outDir = toDirection(theta, phi);
			outPdf = (pdf * float(w*h))/(2.0f*M_PI*M_PI* sinf(theta));
			
			int yIdx = clamp(int(u),0,h-1);
			int xIdx = clamp(int(v),0,w-1);
			//if(yIdx >= h || xIdx >= w)
			//	std::cerr<<"Error: Data acess at "<<xIdx << ":"<<yIdx<<" from "<<u<<" : "<<v<<" at size "<<w<<" x " << h <<"\n";
			return data[yIdx * w + xIdx] ;
		}
	};

	struct light {
		enum type_t { rect, sky };
		type_t type;
		float power;	//!< for light selection, onliy.

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

