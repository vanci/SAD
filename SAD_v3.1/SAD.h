#pragma once
/* Simple Algorithmic Differentiation (SAD)
*/
#include <cassert>

// Memory manager for massively parallel forward mode AD
class MemoryManagerForward
{
public:
	float *D1, *d1; //, *h_d1;
	float *D2, *d2;
	float *dK, *h_dK;
	float *pd;

	int M, N;
	int nvar_S, nvar_K, nvar_T0, nvar_T;
	int nnzpd_S, nnzpd_K, nnzpd_T0, nnzpd_T;

	void Allocate(int M, int N);
	void Clear();
};


// AD session 
class ADS
{
public:
	static int nvar;
	static int nnz_pd;
	static int maxNNZ_pd;
	static int *cooRow, *cooCol;
	static float *pd;

	static void Initialize();
	static void Clear();

	static void GetJacobianForward(float *J, int m, int n);
	static void cudaGetJacobianForward(float *J, int m, MemoryManagerForward &mmf);

	static void ShowNodes();

	static __forceinline void AddNode(int rid, int cid, float v)
	{
		assert(nnz_pd < maxNNZ_pd);
		cooRow[nnz_pd] = rid;
		cooCol[nnz_pd] = cid;
		pd[nnz_pd] = v;
		++nnz_pd;
	}
};

void ShowJacobian(float *J, int m, int n, bool Transposed=false);

// AD variable
// Operations are defined as inline functions to avoid overhead of overloading.
class ADV
{
public:
	int id;
	float v;

	__forceinline ADV& operator= (float a)
	{
		id = ADS::nvar++;
		v = a;
		return *this;
	}
};


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*	Operations supported by SAD
*   Keyword __forceinline is used to guarantee compile-time
*	optimization.
*/

__forceinline ADV Operation1(float lambda, ADV &A, ADV &B)
{
	ADV Y;
	Y.id = ADS::nvar++;
	float pd2 = -lambda/2;
	float pd1 = -3*pd2;
	Y.v = pd1*A.v + pd2*B.v;
	ADS::AddNode(Y.id, A.id, pd1);
	ADS::AddNode(Y.id, B.id, pd2);
	return Y;
}

__forceinline ADV Operation2(float lambda, ADV &A)
{
	ADV Y;
	Y.id = ADS::nvar++;
	float pd = -2*lambda;
	Y.v = 1 + pd*A.v;
	ADS::AddNode(Y.id, A.id, pd);
	return Y;
}

__forceinline ADV Operation3(float lambda, ADV &A, ADV &B)
{
	ADV Y;
	Y.id = ADS::nvar++;
	float pd = lambda/2;
	Y.v = pd*(A.v + B.v);
	ADS::AddNode(Y.id, A.id, pd);
	ADS::AddNode(Y.id, B.id, pd);
	return Y;
}

__forceinline ADV Operation4(float lambda, ADV &A, ADV &B, ADV &C)
{
	ADV Y;
	Y.id = ADS::nvar++;
	float pd2 = -lambda;
	float pd1 = pd2/2;
	Y.v = 1 + pd1*(A.v + C.v) + pd2*B.v;
	ADS::AddNode(Y.id, A.id, pd1);
	ADS::AddNode(Y.id, B.id, pd2);
	ADS::AddNode(Y.id, C.id, pd1);
	return Y;
}

__forceinline ADV InnerProd3(ADV *A, ADV *B)
{
	ADV Y;
	Y.id = ADS::nvar++;
	ADV &A0 = A[0], &A1 = A[1], &A2 = A[2];
	ADV &B0 = B[0], &B1 = B[1], &B2 = B[2]; 
	Y.v = A0.v*B0.v + A1.v*B1.v + A2.v*B2.v;
	ADS::AddNode(Y.id, A0.id, B0.v);
	ADS::AddNode(Y.id, A1.id, B1.v);
	ADS::AddNode(Y.id, A2.id, B2.v);
	ADS::AddNode(Y.id, B0.id, A0.v);
	ADS::AddNode(Y.id, B1.id, A1.v);
	ADS::AddNode(Y.id, B2.id, A2.v);
	return Y;
}

__forceinline ADV SquaredError(int m, float *A, ADV *B)
{
	ADV Y;
	Y.id = ADS::nvar++;
	Y.v = 0;
	for(int i = 0; i < m; ++i)
	{
		float e = B[i].v - A[i];
		Y.v += e*e;
		ADS::AddNode(Y.id, B[i].id, 2*e);
	}
	return Y;
}

/*
		Overloaded operations for float data type
*/

__forceinline float Operation1(float lambda, float &A, float &B)
{
	float Y;
	float pd2 = -lambda/2;
	float pd1 = -3*pd2;
	Y = pd1*A + pd2*B;
	return Y;
}

__forceinline float Operation2(float lambda, float &A)
{
	float Y;
	float pd = -2*lambda;
	Y = 1 + pd*A;
	return Y;
}

__forceinline float Operation3(float lambda, float &A, float &B)
{
	float Y;
	float pd = lambda/2;
	Y = pd*(A + B);
	return Y;
}

__forceinline float Operation4(float lambda, float &A, float &B, float &C)
{
	float Y;
	float pd2 = -lambda;
	float pd1 = pd2/2;
	Y = 1 + pd1*(A + C) + pd2*B;
	return Y;
}

__forceinline float InnerProd3(float *A, float *B)
{
	float Y;
	Y = A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
	return Y;
}