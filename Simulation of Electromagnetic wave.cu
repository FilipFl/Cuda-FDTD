
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define IE 64
//#define JE 64
#define CONFIG 0 //1 = autopilot
#define SHOWIT 0 //1 = wypisywanie
cudaError_t CUDA_INIT_BIG(float* GPUdz, float* GPUhx, float* GPUhy, float* GPUihx, float* GPUihy, float* GPUga);
cudaError_t CUDA_INIT_SMALL(float* GPUgi2, float* GPUgi3, float* GPUfi1, float* GPUfi2, float* GPUfi3, float* GPUgj2, float* GPUgj3, float* GPUfj1, float* GPUfj2, float* GPUfj3);


__global__ void KERNEL_BIG_INIT(float *dz, float *hx, float *hy, float *ihx, float *ihy, float *ga)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int good_j;
	if (j > 0 && j < 4) 
	{
		good_j = j;
		dz[good_j * 1024 + i] = 0.0;
	}
	else if (j > 3 && j < 8) 
	{
		good_j = j - 4;
		hx[good_j * 1024 + i] = 0.0;

	}
	else if (j > 7 && j < 12) 
	{
		good_j = j - 8;
		hy[good_j * 1024 + i] = 0.0;
	}
	else if (j > 11 && j < 16)
	{
		good_j = j - 12;
		ihx[good_j * 1024 + i] = 0.0;
	}
	else if (j > 15 && j < 20)
	{
		good_j = j - 16;
		ihy[good_j * 1024 + i] = 0.0;
	}
	else
	{
		good_j = j - 20;
		ga[good_j * 1024 + i] = 1.0;
	}
}
__global__ void KERNEL_SMALL_INIT(float *gi2, float *gi3, float *fi1, float *fi2, float *fi3, float *gj2, float *gj3, float *fj1, float *fj2, float *fj3)
{
	int i = threadIdx.x;
	int tab = i / 64;
	printf("%d \n", tab);
	int index;
	if (tab == 0) 
	{
		index = i;
		gi2[i] = 1.0f;
	}
	else if (tab == 1) 
	{
		index = i - (tab * 64);
		gi3[index] = 1.0f;
	}
	else if (tab == 2)
	{
		index = i - (tab * 64);
		fi1[index] = 0.0f;
	}
	else if (tab == 3)
	{
		index = i - (tab * 64);
		fi2[index] = 1.0f;
	}
	else if (tab == 4)
	{
		index = i - (tab * 64);
		fi3[index] = 1.0f;
	}
	else if (tab == 5)
	{
		index = i - (tab * 64);
		gj2[index] = 1.0f;
	}
	else if (tab == 6)
	{
		index = i - (tab * 64);
		gj3[index] = 1.0f;
	}
	else if (tab == 7)
	{
		index = i - (tab * 64);
		fj1[index] = 0.0f;
	}
	else if (tab == 8)
	{
		index = i - (tab * 64);
		fj2[index] = 1.0f;
	}
	else if (tab == 9)
	{
		index = i - (tab * 64);
		fj3[index] = 1.0f;
	}
}

void accuracy(float **Matrix_1, float *Matrix_2, float **Matrix_result, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			Matrix_result[i][j] = Matrix_1[i][j] - Matrix_2[i*size + j];
			Matrix_result[i][j] = Matrix_result[i][j] / Matrix_1[i][j];
		}
	}
}

float search(float *Matrix[], int size)
{

	float biggest = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (Matrix[i][j] > biggest)biggest = Matrix[i][j];
		}
	}
	return biggest;
}



int main()
{
	//CPU DECLARATIONS
	float ga[IE][JE], dz[IE][JE], ez[IE][JE], hx[IE][JE], hy[IE][JE];
	int l, n, i, j, ic, jc, nsteps, npml;
	float ddx, dt, T, epsz, pi, epsilon, sigma, eaf;
	float xn, xxn, xnum, xd, curl_e;
	float t0, spread, pulse;
	float gi2[IE], gi3[IE];
	float gj2[JE], gj3[IE];
	float fi1[IE], fi2[IE], fi3[JE];
	float fj1[JE], fj2[JE], fj3[JE];
	float ihx[IE][JE], ihy[IE][JE];
	float matrix_result[IE][JE];
	//GPU DECLARATIONS
	float* GPUga = new float[IE*JE];
	float* GPUdz = new float[IE*JE];
	float* GPUez = new float[IE*JE];
	float* GPUhx = new float[IE*JE];
	float* GPUhy = new float[IE*JE];
	float* GPUgi2 = new float[IE];
	float* GPUgi3 = new float[IE];
	float* GPUgj2 = new float[JE];
	float* GPUgj3 = new float[IE];
	float* GPUfi1 = new float[IE];
	float* GPUfi2 = new float[IE];
	float* GPUfi3 = new float[JE];
	float* GPUfj1 = new float[JE];
	float* GPUfj2 = new float[JE];
	float* GPUfj3 = new float[JE];
	float* GPUihx = new float[IE*JE];
	float* GPUihy = new float[IE*JE];
	float acc = 0;






	FILE *fp;
	
	ic = IE / 2 - 5;
	jc = JE / 2 - 5;
	ddx = .01;		//cell size
	dt = ddx / 6e8;		//time step
	epsz = 8.8e-12;
	pi = 3.14159;
	
	for (j = 0; j < JE; j++) 
	{
		if(SHOWIT ==1)printf("%2d ", j);
		for (i = 0; i < IE; i++) 
		{
			dz[i][j] = 0.0;
			hx[i][j] = 0.0;
			hy[i][j] = 0.0;
			ihx[i][j] = 0.0;
			ihy[i][j] = 0.0;
			ga[i][j] = 1.0;
		if(SHOWIT==1)printf("%5.2f ", ga[i][j]);
		}
		if(SHOWIT==1)printf("\n");
	}

	for (i = 0; i < IE; i++) 
	{
		gi2[i] = 1.0;
		gi3[i] = 1.0;
		fi1[i] = 0.0;
		fi2[i] = 1.0;
		fi3[i] = 1.0;
	}

	for (j = 0; j < IE; j++) 
	{
		gj2[j] = 1.0;
		gj3[j] = 1.0;
		fj1[j] = 0.0;
		fj2[j] = 1.0;
		fj3[j] = 1.0;
	}
	CUDA_INIT_BIG(GPUdz, GPUhx, GPUhy, GPUihx, GPUihy, GPUga);
	CUDA_INIT_SMALL(GPUgi2, GPUgi3, GPUfi1, GPUfi2, GPUfi3, GPUgj2, GPUgj3, GPUfj1, GPUfj2, GPUfj3);
	//accuracy(ga, GPUga, matrix_result, IE);
	printf("%f  \n", GPUga[15 * 64 + 15]);
	printf("%f  \n", ga[15][15]);


	//if (CONFIG == 0) 
	{
		printf("Number of PML cells --> ");
		scanf("%d", &npml);
	}
	if (CONFIG == 1)
	{
		npml = 8;
		printf("Number of PML cells = 8 ");
	}
	for (i = 0; i <= npml; i++) 
	{
		xnum = npml - i;
		xd = npml;
		xxn = xnum / xd;
		xn = 0.33*pow(xxn, 3.0);
		if(SHOWIT==1)printf("%d %7.4f %7.4f \n", i, xxn, xn);
			gi2[i] = 1.0 / (1.0 + xn);
			gi2[IE - 1 - i] = 1.0 / (1.0 + xn);
			gi3[i] = (1.0 - xn) / (1.0 + xn);
			gi3[IE-i-1] = (1.0 - xn) / (1.0 + xn);
			GPUgi2[i] = 1.0 / (1.0 + xn);
			GPUgi2[IE - 1 - i] = 1.0 / (1.0 + xn);
			GPUgi3[i] = (1.0 - xn) / (1.0 + xn);
			GPUgi3[IE - i - 1] = (1.0 - xn) / (1.0 + xn);
		xxn = (xnum - .5) / xd;
		xn = 0.25*pow(xxn, 3.0);
			GPUfi1[i] = xn;
			GPUfi1[IE - 2 - i] = xn;
			GPUfi2[i] = 1.0 / (1.0 + xn);
			GPUfi2[IE - 2 - i] = 1.0 / (1.0 + xn);
			GPUfi3[i] = (1.0 - xn) / (1.0 + xn);
			GPUfi3[IE - 2 - i] = (1.0 - xn) / (1.0 + xn);
	}	
	for (j = 0; j <= npml; j++)
	{
		xnum = npml - j;
		xd = npml;
		xxn = xnum / xd;
		xn = 0.33*pow(xxn, 3.0);
		if(SHOWIT==1)printf("%d %7.4f %7.4f \n", j, xxn, xn);
		gj2[j] = 1.0 / (1.0 + xn);
		gj2[JE - 1 - j] = 1.0 / (1.0 + xn);
		gj3[j] = (1.0 - xn) / (1.0 + xn);
		gj3[JE - j - 1] = (1.0 - xn) / (1.0 + xn);
		GPUgj2[j] = 1.0 / (1.0 + xn);
		GPUgj2[JE - 1 - j] = 1.0 / (1.0 + xn);
		GPUgj3[j] = (1.0 - xn) / (1.0 + xn);
		GPUgj3[JE - j - 1] = (1.0 - xn) / (1.0 + xn);
		xxn = (xnum - .5) / xd;
		xn = 0.25*pow(xxn, 3.0);
		fj1[j] = xn;
		fj1[JE - 2 - j] = xn;
		fj2[j] = 1.0 / (1.0 + xn);
		fj2[JE - 2 - j] = 1.0 / (1.0 + xn);
		fj3[j] = (1.0 - xn) / (1.0 + xn);
		fj3[JE - 2 - j] = (1.0 - xn) / (1.0 + xn);
		GPUfj1[j] = xn;
		GPUfj1[JE - 2 - j] = xn;
		GPUfj2[j] = 1.0 / (1.0 + xn);
		GPUfj2[JE - 2 - j] = 1.0 / (1.0 + xn);
		GPUfj3[j] = (1.0 - xn) / (1.0 + xn);
		GPUfj3[JE - 2 - j] = (1.0 - xn) / (1.0 + xn);
	}
	if (SHOWIT == 1)printf("gi + fi \n");
	for (i = 0; i < IE; i++) 
	{
		if (SHOWIT == 1)printf("%2d   %5.2f  %5.2f \n", i, gi2[i], gi3[i]);
		if (SHOWIT == 1)printf("%t.2f   %5.2f  %5.2f \n", fi1[i], fi2[i], fi3[i]);
	}
	if (SHOWIT == 1)printf("gj + fj \n");
	for (j = 0; j < JE; j++)
	{
		if (SHOWIT == 1)printf("%2d   %5.2f  %5.2f \n", j, gj2[j], gj3[j]);
		if (SHOWIT == 1)printf("%t.2f   %5.2f  %5.2f \n", fj1[j], fj2[j], fj3[j]);
	}
	
	t0 = 40.0;
	spread = 15.0;
	T = 0;
	nsteps = 1;

	while (nsteps > 0)
	{
		if (CONFIG == 0)
		{
			printf("nsteps --> ");
			scanf("%d", &nsteps);
			printf("%d \n", nsteps);
		}
		if (CONFIG == 1)
		{
			nsteps = 50;
			printf("%d \n", nsteps);
		}

		for (n = 1; n < nsteps; n++) {
			T = T + 1;
			//dz field
			for (j = 1; j < IE; j++)
			{
				for (i = 1; i < IE; i++)
				{
					dz[i][j] = gi3[i] * gj3[j] * dz[i][j] + gi2[i] * gj2[j] * .5*(hy[i][j] - hy[i - 1][j] - hx[i][j] + hx[i][j - 1]);
				}
			}
			//sinusoidal source

			pulse = sin(2 * pi * 1500 * 1e6*dt*T);
			dz[ic][jc] = pulse;

			//EZ field

			for (j = 0; j < JE; j++)
			{
				for (i = 0; i < IE; i++)
				{
					ez[i][j] = ga[i][j] * dz[i][j];
				}

			}
			if (SHOWIT == 1)printf("%f %6.2  \n", T, ez[ic][jc]);

			//edges = 0
			for (j = 0; j < JE - 1; j++)
			{
				ez[0][j] = 0.0;
				ez[IE - 1][j] = 0.0;
			}
			for (i = 0; i < IE - 1; i++)
			{
				ez[i][0] = 0.0;
				ez[i][JE - 1] = 0.0;
			}

			//hx field

			for (j = 0; j < JE - 1; j++)
			{
				for (i = 0; i < IE; i++)
				{
					curl_e = ez[i][j] - ez[i][j + 1];
					ihx[i][j] = ihx[i][j] + fi1[i] * curl_e;
					hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * .5*(curl_e + ihx[i][j]);
				}
			}
			//hy field
			for (j = 0; j <= JE - 1; j++)
			{
				for (i = 0; i < IE-1; i++)
				{
					curl_e = ez[i+1][j] - ez[i][j];
					ihy[i][j] = ihy[i][j] + fj1[i] * curl_e;
					hy[i][j] = fi3[j] * hy[i][j] + fi2[j] * .5*(curl_e + ihy[i][j]);
				}
			}

			for (j = 1; j < JE; j++) 
			{
				if (SHOWIT == 1)printf("%2d ", j);
				for (i = 1; i <= IE; i++) 
				{
					if (SHOWIT == 1)printf("%4.lf", ez[i][j]);
				}
				if (SHOWIT == 1)printf(" \n");
			}
			fp = fopen( "Ez.txt" , "w");
			for (j = 0; j < JE; j++) 
			{
				for (i = 0; i < IE; i++) 
				{
					fprintf(fp,"  %0.3f  ", ez[i][j]);
				}
				fprintf(fp, " \n");
			}
			fclose(fp);
			if (SHOWIT == 1)printf(" T = %6.0f \n", T);
			Sleep(3);
		}
	}



	//MEM RELEASE
	delete[] GPUga;
	delete[] GPUdz;
	delete[] GPUez;
	delete[] GPUhx;
	delete[] GPUhy;
	delete[] GPUgi2;
	delete[] GPUgi3;
	delete[] GPUgj2;
	delete[] GPUgj3;
	delete[] GPUfi1;
	delete[] GPUfi2;
	delete[] GPUfi3;
	delete[] GPUfj1;
	delete[] GPUfj2;
	delete[] GPUfj3;
	delete[] GPUihx;
	delete[] GPUihy;

	return 0;
}


cudaError_t CUDA_DZ_FIELD(float* GPUdz, float* GPUgi3, float* GPUgj3,float *GPUgi2, float *GPUgj2,float *GPUhx,float *GPUhy) 
{
	float *dev_dz = 0;
	float *dev_gi3 = 0;
	float *dev_gj3 = 0;
	float *dev_gi2 = 0;
	float *dev_gj2 = 0;
	float *dev_hx = 0;
	float *dev_hy = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_dz, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}




Error:
	cudaFree(dev_dz);
	cudaFree(dev_hx);
	cudaFree(dev_hy);
	cudaFree(dev_gi3);
	cudaFree(dev_gj3);
	cudaFree(dev_gi2);
	cudaFree(dev_gj2);

	return cudaStatus;

}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t CUDA_INIT_BIG(float* GPUdz, float* GPUhx, float* GPUhy, float* GPUihx, float* GPUihy, float* GPUga) 
{
	float *dev_dz = 0;
	float *dev_hx = 0;
	float *dev_hy = 0;
	float *dev_ihx = 0;
	float *dev_ihy = 0;
	float *dev_ga = 0;
	
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_dz, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_hx, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_hy, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ihx, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ihy, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ga, IE * JE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	KERNEL_BIG_INIT <<<24, 1024 >>> (dev_dz, dev_hx, dev_hy, dev_ihx, dev_ihy, dev_ga );
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(GPUdz, dev_dz, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUhx, dev_hx, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}cudaStatus = cudaMemcpy(GPUhy, dev_hy, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}cudaStatus = cudaMemcpy(GPUihx, dev_ihx, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}cudaStatus = cudaMemcpy(GPUihy, dev_ihy, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}cudaStatus = cudaMemcpy(GPUga, dev_ga, IE *JE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_dz);
	cudaFree(dev_hx);
	cudaFree(dev_hy);
	cudaFree(dev_ihx);
	cudaFree(dev_ihy);
	cudaFree(dev_ga);

	return cudaStatus;
}
cudaError_t CUDA_INIT_SMALL(float* GPUgi2, float* GPUgi3, float* GPUfi1, float* GPUfi2, float* GPUfi3, float* GPUgj2, float* GPUgj3, float* GPUfj1, float* GPUfj2, float* GPUfj3) 
{
	float *dev_gi2 = 0;
	float *dev_gi3 = 0;
	float *dev_fi1 = 0;
	float *dev_fi2 = 0;
	float *dev_fi3 = 0;
	float *dev_fj1 = 0;
	float *dev_fj2 = 0;
	float *dev_fj3 = 0;
	float *dev_gj2 = 0;
	float *dev_gj3 = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_gi2, IE  * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_gi3, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fi1, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fi2, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fi3, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fj1, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fj2, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_fj3, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_gj2, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_gj3, IE * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	KERNEL_SMALL_INIT << <1, 1024 >> > (GPUgi2, GPUgi3, GPUfi1, GPUfi2, GPUfi3, GPUgj2, GPUgj3, GPUfj1, GPUfj2, GPUfj3);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching KERNEL_SMALL_INIT!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUgi2, dev_gi2, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUgi3, dev_gi3, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfi1, dev_fi1, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfi2, dev_fi2, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfi3, dev_gi3, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUgj2, dev_gj2, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUgj3, dev_gj3, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfj1, dev_fj1, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfj2, dev_fj2, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(GPUfj3, dev_fj3, IE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_gi2);
	cudaFree(dev_gi3);
	cudaFree(dev_fi1);
	cudaFree(dev_fi2);
	cudaFree(dev_fi3);
	cudaFree(dev_fj1);
	cudaFree(dev_fj2);
	cudaFree(dev_fj3);
	cudaFree(dev_gj2);
	cudaFree(dev_gj3);

	return cudaStatus;
}

/*
cudaError_t PiWithCuda(double *serie, double*sum) 
{
	clock_t start, end;
	double cpu_time_used;
	double gpu_time_used;
	double *dev_sum;
	double *dev_sum_grid;
	double *dev_sum_2;
	double good_Pi = 3.14159265358979323846;
	int liczbawatkow = 1024;
	int liczbablokow = floor((steps)/liczbawatkow);
	//printf("liczba blokow %d \n",liczbablokow);
	double* out = new double[steps];
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_sum, steps*sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_sum_2, steps * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_sum_grid, steps * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_sum, serie,	steps*sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	start = clock();
	PiKernel <<<liczbablokow, liczbawatkow, 2048*sizeof(double) >> > ( dev_sum,dev_sum_2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PiKernel <<<1, liczbawatkow, 2048 * sizeof(double) >> > (dev_sum_2,dev_sum_grid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	end = clock();
	cudaMemcpy(out, dev_sum_grid, sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	double gpi = sqrt(6 * out[0]);
	printf("GPi jest rowne %.20lf \n", gpi);
	double accuracy_cpu = (good_Pi - gpi) / good_Pi;
	printf("Blad GPU: %.12e \n", accuracy_cpu);

	printf("Cuda skonczyla liczyc! \n Zajelo jej to %f sekund \n", gpu_time_used);
Error:
	cudaFree(dev_sum);
	cudaFree(dev_sum_grid);

	

	return cudaStatus;
}
*/