#include <iostream>

using namespace std;

unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)  
{  
	unsigned b=(((z << S1) ^ z) >> S2);  
	return z = (((z & M) << S3) ^ b);  
}  

unsigned LCGStep(unsigned &z, unsigned A, unsigned C)  
{  
	return z=(A*z+C);  
}  

unsigned int z1, z2, z3, z4;
float hybrid() {
	  // Combined period is lcm(p1,p2,p3,p4)~ 2^121  
	return 2.3283064365387e-10 * // Periods  
		(TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1  
		 TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1  
		 TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1  
		 LCGStep(z4, 1664525, 1013904223UL));        // p4=2^32  
}

int main() {
	cout << "unitsize(.1cm);" << endl;
	cout << "real y=0;" << endl;
	cout << "draw((0,y)--(100,y)--(100,y+100)--(0,y+100)--cycle);" << endl;
	int N = 1000;
	for (int i = 0; i < N; ++i) {
		float x = hybrid();
		float y = hybrid();
		cout << "dot((" << x << "*100, " << y << "*100),hsv("<<i<<"*.36,.8,.8));" << endl;
	}
	cout << "y=-110;" << endl;
	z1 = z2 = z3 = z4 = 0xd9f7f;
	cout << "draw((0,y)--(100,y)--(100,y+100)--(0,y+100)--cycle);" << endl;
	for (int i = 0; i < N; ++i) {
		float x = 2.3283064365387e-10 *TausStep(z1, 13,19,12,4294967280UL);
		float y = 2.3283064365387e-10 *TausStep(z1, 13,19,12,4294967280UL);
		cout << "dot((" << x << "*100, " << y << "*100+y),hsv("<<i<<"*.36,.8,.8));" << endl;
	}
	cout << "y=-220;" << endl;
	cout << "draw((0,y)--(100,y)--(100,y+100)--(0,y+100)--cycle);" << endl;
	for (int i = 0; i < N; ++i) {
		float x = 2.3283064365387e-10 *TausStep(z2, 2,25,4,4294967288UL);
		float y = 2.3283064365387e-10 *TausStep(z2, 2,25,4,4294967288UL);
		cout << "dot((" << x << "*100, " << y << "*100+y),hsv("<<i<<"*.36,.8,.8));" << endl;
	}
	cout << "y=-330;" << endl;
	cout << "draw((0,y)--(100,y)--(100,y+100)--(0,y+100)--cycle);" << endl;
	for (int i = 0; i < N; ++i) {
		float x = 2.3283064365387e-10 *TausStep(z3, 3,11,17,4294967280UL);
		float y = 2.3283064365387e-10 *TausStep(z3, 3,11,17,4294967280UL);
		cout << "dot((" << x << "*100, " << y << "*100+y),hsv("<<i<<"*.36,.8,.8));" << endl;
	}
	cout << "y=-440;" << endl;
	cout << "draw((0,y)--(100,y)--(100,y+100)--(0,y+100)--cycle);" << endl;
	for (int i = 0; i < N; ++i) {
		float x = 2.3283064365387e-10 *LCGStep(z4, 1664525, 1013904223UL);
		float y = 2.3283064365387e-10 *LCGStep(z4, 1664525, 1013904223UL);
		cout << "dot((" << x << "*100, " << y << "*100+y),hsv("<<i<<"*.36,.8,.8));" << endl;
	}
}

/* vim: set foldmethod=marker: */

