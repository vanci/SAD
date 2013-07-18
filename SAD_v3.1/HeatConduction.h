#pragma once
#include "SAD.h"

class Rod
{
public:
	static float L;
	static float T;
	static int M;
	static int N;
	static float lambda;
	static float *t0;

	float *s;
	float *tN;

	static void Initialize(int M);
	static void Clear();
	static Rod CreateRod(float *s =0, float *tN =0);
	void Destroy();
	void Show(bool isTerminal=true);
};

template <class Type>
void Solve_HeatEquation(Type *s, Type *tN)
	{
	// Prepare constants and allocate memory for variables
	int M = Rod::M;
	int N = Rod::N;
	float lambda = Rod::lambda;

	Type *K = new Type[(M-1)*3], *pK;
	K[0] = Operation1(lambda, s[0], s[1]);
	K[1] = Operation2(lambda, s[0]);
	K[2] = Operation3(lambda, s[0], s[1]);
	int c = M-2;
	for(int i = 1; i < c; ++i)
	{
		pK = K + i*3;
		pK[0] = Operation3(lambda, s[i-1], s[i]);
		pK[1] = Operation4(lambda, s[i-1], s[i], s[i+1]);
		pK[2] = Operation3(lambda, s[i], s[i+1]);
	}
	pK = K + (M-2)*3;
	pK[0] = Operation3(lambda, s[M-3], s[M-2]);
	pK[1] = Operation2(lambda, s[M-2]);
	pK[2] = Operation1(lambda, s[M-2], s[M-3]);

	Type *ta, *tb, *tc;
	ta = new Type[M+1];
	tb = new Type[M+1];
	ta[0] = Rod::t0[0];
	tb[0] = Rod::t0[0];
	for(int i = 1; i < M; ++i)
		ta[i] = Rod::t0[i];
	ta[M] = Rod::t0[M];
	tb[M] = Rod::t0[M];

	for(int n = 0; n < N; ++n)
	{
		for(int j = 1; j < M; ++j)
		{
			pK = K + (j-1)*3;
			tb[j] = InnerProd3(pK, ta+j-1);
		}
		tc = tb;
		tb = ta;
		ta = tc;
	}

	for(int j = 1; j < M; ++j)
	{
		tN[j] = ta[j];
	}

	delete [] K;
	delete [] ta;
	delete [] tb;
}

