#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "fir.h"

// shmem
static const int buflen = 100e6;
extern char ioRaw[buflen];
extern short int raw[2][buflen / 4];
extern short int rawo[2][buflen / 4];
extern Fir lFir;

__global__ void addKernel(short int* c, const short int* a,
	const short int* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

const int blocksize = 1024;
static const int depth = Fir::depth;
static const int taps = 1 << depth;

__global__ void convolKernel(short int* y, const short int u[], const long long k[], const int repeat) {
	int j = threadIdx.x;
	long long aa[blocksize];
//	__shared__ long long kk[taps];

//	memcpy(kk, k, taps * sizeof(kk[0]));
	
	for (int ii = 0; ii < repeat; ii++) {
		aa[j] = 0;
#pragma unroll taps
		for (int i = 0; i < taps; i++) {
			aa[j] += (0 //
				+ u[ii * blocksize + j + i + 0] * k[i + 0] //
				);
		}

		y[ii * blocksize + j] = aa[j] >> 64 - 16;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda() // short int* c, const short int* a, const short int* b, unsigned int size)
{
	//&rawo[0][ptr], & raw[0][ptr], & raw[1][ptr], blocksize
	int size_uy = 25e6;
	int size_ak = 1 << depth;

	short int* dev_u[2];
	long long* dev_k = 0;
	long long* dev_a = 0;
	short int* dev_y[2];
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
			"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_y[0], size_uy * sizeof(dev_y[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_y[1], size_uy * sizeof(dev_y[1][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_u[0], size_uy * sizeof(dev_u[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_u[1], size_uy * sizeof(dev_u[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_k, size_ak * sizeof(dev_k[0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size_ak * sizeof(dev_a[0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_u[0], raw[0], size_uy * sizeof(dev_u[0][0]),
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_u[1], raw[1], size_uy * sizeof(dev_u[1][0]),
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_k, lFir.k, size_ak * sizeof(dev_k[0]),
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	{	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << < 1, size >> > (dev_c, dev_a, dev_b);

			convolKernel << < 1, blocksize >> > (&dev_y[0][0], &dev_u[0][0], dev_k, 12100);
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n",
					cudaGetErrorString(cudaStatus));
				goto Error;
			}
			
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr,
					"cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
					cudaStatus);
				goto Error;
			}
			
			convolKernel << < 1, blocksize >> > (&dev_y[1][0], &dev_u[1][0], dev_k, 12100);
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n",
					cudaGetErrorString(cudaStatus));
				goto Error;
			}
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
			"cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
			cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(rawo[0], dev_y[0], size_uy * sizeof(rawo[0][0]),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(rawo[1], dev_y[1], size_uy * sizeof(rawo[1][0]),
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error: 
	cudaFree(dev_y[0]);
	cudaFree(dev_y[1]);
	cudaFree(dev_u[0]);
	cudaFree(dev_u[1]);
	cudaFree(dev_k);
	cudaFree(dev_a);

	return cudaStatus;
}

int maine() {
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(); //&rawo[0][ptr], &raw[0][ptr], &raw[1][ptr], blocksize
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	// NOTE: C-K C-F for "Format Selection"

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

