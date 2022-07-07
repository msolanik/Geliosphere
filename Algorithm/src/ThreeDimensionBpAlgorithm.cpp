#include "ThreeDimensionBpAlgorithm.hpp"

#include "ThreeDimensionBpCpuSimulation.hpp"
#include "TwoDimensionBpResults.hpp"

void ThreeDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{

	}
	else
	{
		ThreeDimensionBpCpuSimulation *threeDimensionBpCpuSimulation = new ThreeDimensionBpCpuSimulation();
		threeDimensionBpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);
}

// Compute capability actual device
void ThreeDimensionBpAlgorithm::setThreadBlockSize()
{
	// cudaDeviceProp gpuProperties;
	// gpuErrchk(cudaGetDeviceProperties(&gpuProperties, 0));
	// int computeCapability = gpuProperties.major * 100 + gpuProperties.minor * 10;
	// switch (computeCapability)
	// {
	// case 610:
	// 	blockSize = 65536;
	// 	threadSize = 512;
	// 	break;
	// case 750:
	// 	blockSize = 32768;
	// 	threadSize = 1024;
	// 	break;
	// default:
	// 	blockSize = 64;
	// 	threadSize = 64;
	// 	break;
	// }
}