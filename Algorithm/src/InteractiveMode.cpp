#include "InteractiveMode.hpp"
#include "CudaErrorCheck.cuh"

InteractiveMode::InteractiveMode()
{
	printf("Initializing RNG");
	setThreadBlockSize();
    gpuErrchk(cudaMallocManaged(&state, ((blockSize * threadSize) * sizeof(curandState_t))));
    initRNG(state, blockSize, threadSize);
	printf("Initializing RNG");
}

InteractiveMode::~InteractiveMode()
{
	gpuErrchk(cudaFree(state));
}
    
curandState_t * InteractiveMode::getRNGStateStructure()
{
    return state;
}

void InteractiveMode::setThreadBlockSize()
{
	cudaDeviceProp gpuProperties;
	gpuErrchk(cudaGetDeviceProperties(&gpuProperties, 0));
	int computeCapability = gpuProperties.major * 100 + gpuProperties.minor * 10;
	switch (computeCapability)
	{
	case 610:
		blockSize = 32768;
		threadSize = 512;
		break;
	case 750:
		blockSize = 16384;
		threadSize = 1024;
		break;
	default:
		blockSize = 64;
		threadSize = 64;
		break;
	}
}