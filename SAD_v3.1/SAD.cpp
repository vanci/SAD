#include <cstdio>
#include <memory>
#include <cassert>

#include "SAD.h"

int ADS::nvar = 0;
int ADS::nnz_pd = 0;
int ADS::maxNNZ_pd = 0;
int* ADS::cooRow = 0;
int* ADS::cooCol = 0;
float* ADS::pd = 0;

void ADS::Initialize()
{
	Clear();
	nvar = 0;
	nnz_pd = 0;
	maxNNZ_pd = (1<<28);
	cooRow = new int[maxNNZ_pd];
	cooCol = new int[maxNNZ_pd];
	pd = new float[maxNNZ_pd];
}

void ADS::Clear()
{
	if(cooRow)
	{
		delete [] cooRow;
		delete [] cooCol;
		delete [] pd;
	}
	cooRow = 0;
	cooCol = 0;
	pd = 0;
}

void ADS::ShowNodes()
{
	for(int i = 0; i < nnz_pd; ++i)
	{
		printf("(%2d, %2d)\t%6.2f\n", cooRow[i], cooCol[i], pd[i]);
	}
	putchar('\n');
}

void ShowJacobian(float *J, int m, int n, bool Transposed)
{
	for(int i = 0; i < m; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			if(Transposed)
			{
				printf("%6.4f\t", J[i*n + j] );
			}
			else{
				printf("%6.4f\t", J[i+j*m] );
			}
		}
		putchar('\n');
	}
}

void ADS::GetJacobianForward(float *J, int m, int n)
{
	float *adj = new float[ADS::nvar];
	memset(adj, 0, sizeof(*adj)*ADS::nvar);

	int *rid = ADS::cooRow;
	int *cid = ADS::cooCol;
	float *pd = ADS::pd;

	for(int xid = 0; xid < n; ++xid)
	{
		adj[xid] = 1;
		for(int j = 0; j < ADS::nnz_pd; ++j)
		{
			adj[ rid[j] ] += pd[j]*adj[cid[j]];
		}
		memcpy(J+m*xid, adj + ADS::nvar - m, sizeof(*J)*m);
		memset(adj, 0, sizeof(*adj)*ADS::nvar);
	}
	delete [] adj;
}