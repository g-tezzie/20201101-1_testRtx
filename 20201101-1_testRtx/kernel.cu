/*
 *
 * from: https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
 *
 *Enabling certain metrics can cause GPU kernels to run longer than the driver's watchdog time-out limit. In these cases the driver will terminate the GPU kernel resulting in an application error and profiling data will not be available. Please disable the driver watchdog time out before profiling such long running CUDA kernels.
 On Linux, setting the X Config option Interactive to false is recommended.
 For Windows, detailed information on disabling the Windows TDR is available at https://docs.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math_constants.h"
//#include "math.h"

#include <stdio.h>
#include <iostream>

#include "fir.h"

// shmem
static const int buflen = 100e6;
extern char ioRaw[buflen];
extern short int raw[2][buflen / 4];
extern short int rawo[2][buflen / 4];
extern Fir lFir;

const int blocksize = 1024;
static const int depth = Fir::depth;
static const int taps = 1 << depth;

__device__ float core_sinc(int i) {

	static const float fl_ = 200.0 / 48000.0;
	const float pi = CUDART_PI; // atan2(-1, 0);
	float wl = fl_ * 2 * pi;
	float no2 = (taps - 1) / 2.0;

	float no2Now = -no2 + i;
	float xn = sin(wl * no2Now);
	float xd = pi * no2Now;
	float invd = 1.0 / xd;
	float xx = xn * invd;	// ‚±‚¤‚È‚ç‚È‚¢‚æ‚¤‚É•ÛØ no2Now == 0 ? 2 * fl : xn / xd;
	return xx;
}

__global__ void convolKernel(short int *y, const short int u[],
		const long long k[], const int repeat) {
	int j = threadIdx.x;
	float aa[blocksize];
	//	__shared__ long long kk[taps];

	//	memcpy(kk, k, taps * sizeof(kk[0]));

	for (int ii = 0; ii < repeat; ii++) {
		aa[j] = 0;
#pragma unroll taps/2
		for (int i = 0; i < taps / 2; i++) {
			aa[j] += (u[ii * blocksize + j + i + 0]
					+ u[ii * blocksize + j + taps - i - 1]) * core_sinc(i);
		}

		y[ii * blocksize + j] = aa[j];
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda() // short int* c, const short int* a, const short int* b, unsigned int size)
{
	//&rawo[0][ptr], & raw[0][ptr], & raw[1][ptr], blocksize
	int size_uy = 25e6;
	int size_ak = 1 << depth;

	short int *dev_u[2];
	long long *dev_k = 0;
	long long *dev_a = 0;
	short int *dev_y[2];
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**) &dev_y[0], size_uy * sizeof(dev_y[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_y[1], size_uy * sizeof(dev_y[1][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_u[0], size_uy * sizeof(dev_u[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_u[1], size_uy * sizeof(dev_u[0][0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_k, size_ak * sizeof(dev_k[0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_a, size_ak * sizeof(dev_a[0]));
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

	Error: cudaFree(dev_y[0]);
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

