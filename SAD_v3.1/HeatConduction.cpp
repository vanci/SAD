#include "HeatConduction.h"
#include <memory>
#include <ctime>

float Rod::L = 0;
float Rod::T = 0;
int Rod::M = 0;
int Rod::N = 0;
float Rod::lambda = 0;
float* Rod::t0 = 0;

void Rod::Initialize(int M)
{
	L = 1;
	T = 0.5;
	Rod::M = M;
	N = M*M;
	lambda = T*M*M/(N*L*L);
	t0 = new float[M+1];
	memset(t0, 0, sizeof(*t0)*(M+1));
	t0[0] = 1;
	t0[M] = 0.5;
}

void Rod::Clear()
{
	if(t0)
		delete [] t0;
}

Rod Rod::CreateRod(float *s, float *tN)
{
	Rod rod;
	rod.s = new float[Rod::M-1];
	if(s)    // conductivity of interior points
	{
		memcpy(rod.s, s, sizeof(*s)*(Rod::M-1));
	}else
	{
		memset(rod.s, 0, sizeof(*s)*(Rod::M-1));
	}

	rod.tN = new float[Rod::M+1];
	if(tN)
	{
		memcpy(rod.tN, tN, sizeof(*tN)*(Rod::M+1));
	}else
	{
		memset(rod.tN, 0, sizeof(*tN)*(Rod::M+1));
	}
	rod.tN[0] = Rod::t0[0];
	rod.tN[Rod::M] = Rod::t0[M];
	return rod;
}

void Rod::Destroy()
{
	delete [] s;
	delete [] tN;
	s = 0;
	tN = 0;
}

void Rod::Show(bool isTerminal)
{
	float *t = t0;
	if( isTerminal )
		t = tN;
	putchar('\n');
	for(int i = 0; i <= M; ++i)
		printf("%6.4f ", t[i]);
	putchar('\n');
}

