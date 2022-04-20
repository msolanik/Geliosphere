#include "TwoDimensionBpAlgorithm.hpp"

#include <cuda_runtime.h>
#include "CudaErrorCheck.cuh"

#include "TwoDimensionBpCpuSimulation.hpp"
#include "TwoDimensionBpResults.hpp"

void TwoDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	// if (!singleTone->getInt("isCpu", 0))
	// {

	// }
	// else
	// {
		TwoDimensionBpCpuSimulation *twoDimensionBpCpuSimulation = new TwoDimensionBpCpuSimulation();
		twoDimensionBpCpuSimulation->runSimulation(singleTone);
	// }

	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);
}

// Compute capability actual device
void TwoDimensionBpAlgorithm::setThreadBlockSize()
{
	cudaDeviceProp gpuProperties;
	gpuErrchk(cudaGetDeviceProperties(&gpuProperties, 0));
	int computeCapability = gpuProperties.major * 100 + gpuProperties.minor * 10;
	switch (computeCapability)
	{
	case 610:
		blockSize = 65536;
		threadSize = 512;
		break;
	case 750:
		blockSize = 32768;
		threadSize = 1024;
		break;
	default:
		blockSize = 64;
		threadSize = 64;
		break;
	}
}