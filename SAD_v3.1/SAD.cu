#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <cassert>

#include "SAD.h"

#define KERNEL_LENGTH	10000000
#define CUDA_CHECK(x,y)  if((x) != cudaSuccess){ puts(y); assert(0); }
#define MIN(a,b)	((a)<(b)?(a):(b))

// MemoryManager allocates memory for Jacobian computation
void MemoryManagerForward::Allocate(int M, int N)
{
	this->M = M;
	this->N = N;

	// Expected number of variables
	nvar_S = M-1;
	nvar_K = nvar_S + 3*(M-1);
	nvar_T0  = nvar_K + (M-1) + 4;
	nvar_T = nvar_T0 + (M-1)*M*M;

	// Expected number of arcs in the DAG
	nnzpd_S = 0;
	nnzpd_K = 5 + (M-3)*7 + 5;
	nnzpd_T0 = nnzpd_K;
	nnzpd_T = nnzpd_T0 + (M-1)*M*M*6;

	assert(ADS::nvar == nvar_T);	// actual number of variables
	assert(ADS::nnz_pd == nnzpd_T);	// actual number of arcs in the DAG

	cudaError_t err = cudaSuccess;

	err = cudaMalloc(&D1, sizeof(*D1)*(1+M*(M-1)));
	CUDA_CHECK(err, "D1 allocation failed.");
	cudaMemset(D1, 0, sizeof(*D1)*(1+M*(M-1)));
	d1 = D1 + 1;

	err = cudaMalloc(&D2, sizeof(*D1)*(1+M*(M-1)));
	CUDA_CHECK(err, "D2 allocation failed.");
	cudaMemset(D2, 0, sizeof(*D1)*(1+M*(M-1)));
	d2 = D2 + 1;

	err = cudaMalloc(&dK, sizeof(*dK)*(nvar_K-nvar_S)*(M-1));
	CUDA_CHECK(err, "dK allocation failed.");
	h_dK = new float[(nvar_K-nvar_S)];
	memset(h_dK, 0, sizeof(*h_dK)*(nvar_K-nvar_S));

	err = cudaMalloc(&pd, sizeof(*pd)*ADS::nnz_pd);
	CUDA_CHECK(err,"pd allocation failed.");
	err = cudaMemcpy(pd, ADS::pd, sizeof(*pd)*ADS::nnz_pd, cudaMemcpyHostToDevice);
	CUDA_CHECK(err,"pd memcpy failed.");
		
}

void MemoryManagerForward::Clear()
{
	cudaFree(D1);
	cudaFree(D2);
	cudaFree(dK);
	cudaFree(pd);

	delete [] h_dK;	
}

__global__ void propagateKernel(float *p1, float *p2, float *dK, float *pd, int M, int N, int b)
{
	int j = threadIdx.x;
	int k = 6*j;

	float *d1 = p1 + blockIdx.x * M;
	float *d2 = p2 + blockIdx.x * M;
	float *t, *result;

	float dK_row[3];
	int m = 3*j + blockIdx.x * 3 * (M-1);
	dK_row[0] = dK[m];
	dK_row[1] = dK[m+1];
	dK_row[2] = dK[m+2];

	for(int n = 0; n < N; ++n)
	{
		d2[j] = pd[k] * dK_row[0] + pd[k+1] * dK_row[1] + pd[k+2] * dK_row[2]
			+ pd[k+3] * d1[j-1] + pd[k+4] * d1[j] + pd[k+5] * d1[j+1];
		k += b;

		t = d2, d2 = d1, d1 = t;
		__syncthreads();
	}
	p1[blockIdx.x * M + j] = d1[j];
}

void ADS::cudaGetJacobianForward(float *J, int m, MemoryManagerForward &mmf)
{
	int M = mmf.M;
	int N = mmf.N;
	int NOI = M - 1;

	cudaError_t err = cudaSuccess;
	float *dK = mmf.dK,
		*h_dK = mmf.h_dK;

	float *d1 = mmf.d1;
	float *d2 = mmf.d2;

	int *rid = ADS::cooRow;
	int *cid = ADS::cooCol;
	float *pd = mmf.pd, *h_pd = ADS::pd;

	for(int xid = 0; xid < NOI; ++xid)
	{
		for(int ipd = 0; ipd < mmf.nnzpd_K; ++ipd)
		{
			if( cid[ipd] == xid )
				h_dK[ rid[ipd] - mmf.nvar_S ] += h_pd[ipd];
		}

		err = cudaMemcpy(dK + xid*3*(M-1), h_dK, sizeof(*h_dK)*3*(M-1), cudaMemcpyHostToDevice);
		CUDA_CHECK(err,"dK memcpy failed.");
		cudaThreadSynchronize();
		memset(h_dK, 0, sizeof(*h_dK)*3*(M-1));
	}

	int C = (M-1)*6;
	int StepSize = KERNEL_LENGTH / C;
	for(int n = 0; n < N; n+=StepSize)
	{
		int n_ceil = MIN(N-n, StepSize);
		propagateKernel<<<m,mmf.nvar_S>>>(d1, d2, dK, pd+mmf.nnzpd_K + C*n, M, n_ceil, C);
		err = cudaThreadSynchronize();
		CUDA_CHECK(err,"Kernel error");
	}

	for(int j = 0; j < NOI; ++j)
	{
		err = cudaMemcpy(J+NOI*j, d1+M*j, sizeof(*J)*NOI, cudaMemcpyDeviceToHost);
		CUDA_CHECK(err,"J memcpy failed.");
	}
}