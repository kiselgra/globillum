#include "simpleMaterial.h"
#include "principledMaterial.h"

using namespace std;
using namespace rta;
using namespace rta::cuda;
using namespace gi;
using namespace gi::cuda;


#define ALL_MATERIAL_LAMBERT 1 
#define ALL_MATERIAL_BLINNPHONG 0


struct materialBRDF{
	bool isSimple;
		#if ALL_MATERIAL_LAMBERT
			LambertianMaterial simple;
		#elif ALL_MATERIAL_BLINNPHONG
			BlinnMaterial simple;
		#endif
		PrincipledMaterial principled;

	void init(bool isPrincipled, const rta::cuda::material_t *mat, const float2 &TC, const float2 &upper_T, const float2 &right_T, const float3 &Tx, const float3 &Ty){
		isSimple = (!isPrincipled);
		if(isSimple) simple.init(mat,TC,upper_T,right_T);
		else principled.init(mat, TC, upper_T, right_T, Tx, Ty);
	}
	void sample(const float3 &inv_org_dir_ts, float3 &dir, const float3 &random, float &pdf){
		if(isSimple) simple.sample(inv_org_dir_ts, dir, random, pdf);
		else principled.sample(inv_org_dir_ts, dir, random, pdf);
	}
	float3 evaluate(const float3 &inv_org_dir, const float3 &light_dir, const float3 &N)
	{
		if(isSimple) return simple.evaluate(inv_org_dir,light_dir,N);
		else return principled.evaluate(inv_org_dir,light_dir,N);
	}	
};
