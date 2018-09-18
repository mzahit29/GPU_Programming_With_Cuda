#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void kernel(int *a, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n)
	{
		a[idx] *= 3;
	}
}

int main()
{
	cout << "main() begin" << endl;
	int N = 100000;
	int size = N * sizeof(int);

	int* a_h{};
	a_h = (int *)malloc(size);
	cout << "after malloc. a_h: " << a_h << endl;
	// Fill host memory with some values
	for (int i = 0; i < N; ++i) a_h[i] = i;

	int block_size = 1024; // Count of threads in a block
	// how many blocks required to handle each element of array with one thread
	// we have set thread count per block to be 4.
	int block_count = N / block_size + ((N % block_size) ? 0 : 1);


	cudaError_t err;

	int* a_d{};
	cudaMalloc(&a_d, size);
	err = cudaGetLastError();
	cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;

	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	cout << "cudaMemcpy host to device: " << cudaGetErrorString(err) << endl;


	// Run the program block on the GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	kernel<<< block_count, block_size >>> (a_d, size);
	cudaEventRecord(stop);

	// Find the calculation time on GPU
	float calc_time = 0;
	cudaEventSynchronize(stop); // Otherwise CPU will not wait for cudaEventElapsedTime to set the calc_time and print 0
	cudaEventElapsedTime(&calc_time, start, stop);
	cout << "GPU calculated the result in " << calc_time << " ms" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// cudaMemcpy is sync therefore it is inherently a synchronization point
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	cout << "cudaMemcpy device to host: " << cudaGetErrorString(err) << endl;
	//cudaDeviceSynchronize();

	// print the result
	//for (int i = 0; i < N; ++i) cout << a_h[i] << endl;

	free(a_h);
	cudaFree(a_d);
	cout << "main() end" << endl;
}