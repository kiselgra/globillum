
void hsvColorMap( float *rgb, float value, float min, float max )
{
	float val = max-value;
	float h = 240.0*val/max;

	float r,g,b;

	float s = 0.9;
	float v = 0.8;

	int hi = floor(h/60.0);
	float f = (h/60.0)-hi;
	float p = v*(1-s);
	float q = v*(1-s*f);
	float t = v*(1-s*(1-f));
	
	
	if (hi == 0 || hi == 6) { r=v;g=t;b=p;}
	else if (hi == 1) { r=q;g=v;b=p;}
	else if (hi == 2) { r=p;g=v;b=t;}  
	else if (hi == 3) { r=p;g=q;b=v;}  
	else if (hi == 4) { r=t;g=p;b=v;}  
	else { r=v;g=p;b=q;}  

	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
	
}
