#include "HeatConduction.h"
#include "SAD.h"

#include <cmath>
#include <cstdio>
#include <ctime>

struct ElapseRecord
{
	int M;
	int N;
	float elap_solve;
	float elap_ad_solve;
	float elap_ad_cpu;
	float elap_ad_gpu;
	float elap_ad_mmf;
	float err;
};

void TestCase(int size, ElapseRecord &r)
{
	Rod::Initialize(size);	// 430 is maximum
	int NOI = Rod::M-1;		// number of interior nodes

	r.M = Rod::M;
	r.N = Rod::N;

	float *s = new float[NOI];
	for(int i = 1; i < Rod::M; ++i)
	{
		s[i-1] = (float)i/(float)Rod::M;	// conductivity at interior nodes
	}

	Rod targetRod = Rod::CreateRod(s);
	clock_t tic = clock();
	Solve_HeatEquation(targetRod.s, targetRod.tN);
	r.elap_solve =  float(clock() - tic)/CLOCKS_PER_SEC;
	
//	puts("Target terminal temperature:");
//	targetRod.Show();

	ADS::Initialize();
	ADV *S = new ADV[Rod::M-1];
	for(int i = 0; i < NOI; ++i)
	{
		S[i] = (float)0.5*s[i];
	}
	
	
	ADV *tN = new ADV[Rod::M+1];
	printf("nvar = %d\tnnz_pd = %d\n", ADS::nvar, ADS::nnz_pd);
	tic = clock();
	Solve_HeatEquation(S, tN);
	r.elap_ad_solve =  float(clock() - tic)/CLOCKS_PER_SEC;
	printf("nvar = %d\tnnz_pd = %d\n", ADS::nvar, ADS::nnz_pd);
	printf("%8.6f sec\t Solve\n", r.elap_solve);
	printf("%8.6f sec\t Solve with AD\n", r.elap_ad_solve);
/*
	puts("\nTerminal temperature:");
	for(int i = 1; i <= NOI; ++i)
		printf("%6.4f ", tN[i].v);
	putchar('\n');
	*/
	
	float *J = new float[NOI*NOI];

	// Get Jacobian using standard forward propagation
	tic = clock();
	ADS::GetJacobianForward(J, Rod::M-1, NOI);
	r.elap_ad_cpu =  float(clock() - tic)/CLOCKS_PER_SEC;
	printf("%8.6f sec\t Gradient by Forward mode\n", r.elap_ad_cpu);
	// Allocate device memory
	tic = clock();
	MemoryManagerForward mmf;
	mmf.Allocate(Rod::M,Rod::N);
	r.elap_ad_mmf =  float(clock() - tic)/CLOCKS_PER_SEC;
	float *J2 = new float[NOI*NOI];
	// Get Jacobian using CUDA forward propagation
	tic = clock();
	ADS::cudaGetJacobianForward(J2, NOI, mmf);
	r.elap_ad_gpu =  float(clock() - tic)/CLOCKS_PER_SEC;
	mmf.Clear();
	printf("%8.6f sec\t Gradient by GPU-Accelerated Forward mode\n", r.elap_ad_gpu);
	printf("%8.6f sec\t CUDA overhead\n\n", r.elap_ad_mmf);

/*	Can't show Jacobian matrix if it's too large.

	puts("\nJacobians obtained by two methods:\n");
	ShowJacobian(J, NOI, NOI, true);
	puts("\n\n");
	ShowJacobian(J2, NOI, NOI, true);
*/

	// Check result
	float max = 0;
	for(int i = NOI*NOI - 1; i >= 0; --i)
	{
		float error = abs(J[i] - J2[i]);
		if( max < error )
			max = error;
	}
	r.err = max;

	printf("Error = %8.6f\n",max);

	delete [] s;
	delete [] S;
	delete [] tN;
	delete [] J;
	delete [] J2;
	targetRod.Destroy();
	ADS::Clear();
	Rod::Clear();
}


int main()
{
	ElapseRecord R[10];
	clock_t begin = clock();

	for(int i = 1; i < 11; ++i)
		TestCase(32*i+1,R[i]);
	clock_t end = clock();

	float elap = float( end - begin ) / CLOCKS_PER_SEC;
	printf("\n\nTotal Elapse = %f sec\n", elap);
	return 0;
}