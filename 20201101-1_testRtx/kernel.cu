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

__global__ void addKernel(short int *c, const short int *a,
		const short int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

static const int depth = 12;
static const int taps = 1 << 12;
__global__ void convolKernel(short int *y, long long *a, const short int u[],
		const long long k[]) {
	int j = threadIdx.x;

	a[j] = 0;
	for (int i = 0; i < taps; i += 128) {
		a[j] += (0 //
		+ u[j + i + 0] * k[i + 0] //
		+ u[j + i + 1] * k[i + 1] //
		+ u[j + i + 2] * k[i + 2] //
		+ u[j + i + 3] * k[i + 3] //
		+ u[j + i + 4] * k[i + 4] //
		+ u[j + i + 5] * k[i + 5] //
		+ u[j + i + 6] * k[i + 6] //
		+ u[j + i + 7] * k[i + 7] //
		+ u[j + i + 8] * k[i + 8] //
		+ u[j + i + 9] * k[i + 9] //
		+ u[j + i + 10] * k[i + 10] //
		+ u[j + i + 11] * k[i + 11] //
		+ u[j + i + 12] * k[i + 12] //
		+ u[j + i + 13] * k[i + 13] //
		+ u[j + i + 14] * k[i + 14] //
		+ u[j + i + 15] * k[i + 15] //
		+ u[j + i + 16] * k[i + 16] //
		+ u[j + i + 17] * k[i + 17] //
		+ u[j + i + 18] * k[i + 18] //
		+ u[j + i + 19] * k[i + 19] //
		+ u[j + i + 20] * k[i + 20] //
		+ u[j + i + 21] * k[i + 21] //
		+ u[j + i + 22] * k[i + 22] //
		+ u[j + i + 23] * k[i + 23] //
		+ u[j + i + 24] * k[i + 24] //
		+ u[j + i + 25] * k[i + 25] //
		+ u[j + i + 26] * k[i + 26] //
		+ u[j + i + 27] * k[i + 27] //
		+ u[j + i + 28] * k[i + 28] //
		+ u[j + i + 29] * k[i + 29] //
		+ u[j + i + 30] * k[i + 30] //
		+ u[j + i + 31] * k[i + 31] //
		+ u[j + i + 32] * k[i + 32] //
		+ u[j + i + 33] * k[i + 33] //
		+ u[j + i + 34] * k[i + 34] //
		+ u[j + i + 35] * k[i + 35] //
		+ u[j + i + 36] * k[i + 36] //
		+ u[j + i + 37] * k[i + 37] //
		+ u[j + i + 38] * k[i + 38] //
		+ u[j + i + 39] * k[i + 39] //
		+ u[j + i + 40] * k[i + 40] //
		+ u[j + i + 41] * k[i + 41] //
		+ u[j + i + 42] * k[i + 42] //
		+ u[j + i + 43] * k[i + 43] //
		+ u[j + i + 44] * k[i + 44] //
		+ u[j + i + 45] * k[i + 45] //
		+ u[j + i + 46] * k[i + 46] //
		+ u[j + i + 47] * k[i + 47] //
		+ u[j + i + 48] * k[i + 48] //
		+ u[j + i + 49] * k[i + 49] //
		+ u[j + i + 50] * k[i + 50] //
		+ u[j + i + 51] * k[i + 51] //
		+ u[j + i + 52] * k[i + 52] //
		+ u[j + i + 53] * k[i + 53] //
		+ u[j + i + 54] * k[i + 54] //
		+ u[j + i + 55] * k[i + 55] //
		+ u[j + i + 56] * k[i + 56] //
		+ u[j + i + 57] * k[i + 57] //
		+ u[j + i + 58] * k[i + 58] //
		+ u[j + i + 59] * k[i + 59] //
		+ u[j + i + 60] * k[i + 60] //
		+ u[j + i + 61] * k[i + 61] //
		+ u[j + i + 62] * k[i + 62] //
		+ u[j + i + 63] * k[i + 63] //
		+ u[j + i + 64] * k[i + 64] //
		+ u[j + i + 65] * k[i + 65] //
		+ u[j + i + 66] * k[i + 66] //
		+ u[j + i + 67] * k[i + 67] //
		+ u[j + i + 68] * k[i + 68] //
		+ u[j + i + 69] * k[i + 69] //
		+ u[j + i + 70] * k[i + 70] //
		+ u[j + i + 71] * k[i + 71] //
		+ u[j + i + 72] * k[i + 72] //
		+ u[j + i + 73] * k[i + 73] //
		+ u[j + i + 74] * k[i + 74] //
		+ u[j + i + 75] * k[i + 75] //
		+ u[j + i + 76] * k[i + 76] //
		+ u[j + i + 77] * k[i + 77] //
		+ u[j + i + 78] * k[i + 78] //
		+ u[j + i + 79] * k[i + 79] //
		+ u[j + i + 80] * k[i + 80] //
		+ u[j + i + 81] * k[i + 81] //
		+ u[j + i + 82] * k[i + 82] //
		+ u[j + i + 83] * k[i + 83] //
		+ u[j + i + 84] * k[i + 84] //
		+ u[j + i + 85] * k[i + 85] //
		+ u[j + i + 86] * k[i + 86] //
		+ u[j + i + 87] * k[i + 87] //
		+ u[j + i + 88] * k[i + 88] //
		+ u[j + i + 89] * k[i + 89] //
		+ u[j + i + 90] * k[i + 90] //
		+ u[j + i + 91] * k[i + 91] //
		+ u[j + i + 92] * k[i + 92] //
		+ u[j + i + 93] * k[i + 93] //
		+ u[j + i + 94] * k[i + 94] //
		+ u[j + i + 95] * k[i + 95] //
		+ u[j + i + 96] * k[i + 96] //
		+ u[j + i + 97] * k[i + 97] //
		+ u[j + i + 98] * k[i + 98] //
		+ u[j + i + 99] * k[i + 99] //
		+ u[j + i + 100] * k[i + 100] //
		+ u[j + i + 101] * k[i + 101] //
		+ u[j + i + 102] * k[i + 102] //
		+ u[j + i + 103] * k[i + 103] //
		+ u[j + i + 104] * k[i + 104] //
		+ u[j + i + 105] * k[i + 105] //
		+ u[j + i + 106] * k[i + 106] //
		+ u[j + i + 107] * k[i + 107] //
		+ u[j + i + 108] * k[i + 108] //
		+ u[j + i + 109] * k[i + 109] //
		+ u[j + i + 110] * k[i + 110] //
		+ u[j + i + 111] * k[i + 111] //
		+ u[j + i + 112] * k[i + 112] //
		+ u[j + i + 113] * k[i + 113] //
		+ u[j + i + 114] * k[i + 114] //
		+ u[j + i + 115] * k[i + 115] //
		+ u[j + i + 116] * k[i + 116] //
		+ u[j + i + 117] * k[i + 117] //
		+ u[j + i + 118] * k[i + 118] //
		+ u[j + i + 119] * k[i + 119] //
		+ u[j + i + 120] * k[i + 120] //
		+ u[j + i + 121] * k[i + 121] //
		+ u[j + i + 122] * k[i + 122] //
		+ u[j + i + 123] * k[i + 123] //
		+ u[j + i + 124] * k[i + 124] //
		+ u[j + i + 125] * k[i + 125] //
		+ u[j + i + 126] * k[i + 126] //
		+ u[j + i + 127] * k[i + 127] //
		);
	}

	y[j] = a[j] >> 64 - 16;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda() // short int* c, const short int* a, const short int* b, unsigned int size)
{
	//&rawo[0][ptr], & raw[0][ptr], & raw[1][ptr], blocksize
	int size_uy = 25e6;
	int size_ak = 1 << depth;

	short int *dev_u = 0;
	long long *dev_k = 0;
	long long *dev_a = 0;
	short int *dev_y = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**) &dev_y, size_uy * sizeof(dev_y[0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**) &dev_u, size_uy * sizeof(dev_u[0]));
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
	cudaStatus = cudaMemcpy(dev_u, raw[0], size_uy * sizeof(dev_u[0]),
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

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << < 1, size >> > (dev_c, dev_a, dev_b);
	const int blocksize = 1024;
	for (int i = 0; i < 10000; i++) {
	convolKernel << < 1, blocksize >> > (&dev_y[blocksize * i], dev_a, &dev_u[blocksize * i], dev_k);
}
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

// Copy output vector from GPU buffer to host memory.
cudaStatus = cudaMemcpy(rawo[0], dev_y, size_uy * sizeof(rawo[0][0]),
		cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
}

Error: cudaFree(dev_y);
cudaFree(dev_u);
cudaFree(dev_k);

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

