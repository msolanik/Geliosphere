#include "OneDimensionBpAlgorithm.hpp"

#include <cuda_runtime.h>
#include "CudaErrorCheck.cuh"
#include "OneDimensionBpResults.hpp"

OneDimensionBpAlgorithm::OneDimensionBpAlgorithm(InteractiveMode *interactiveMode)
{
	this->interactiveMode = interactiveMode;
}

void OneDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	simulationInputBP simulation;
	setThreadBlockSize();
	curandState_t *state;
	double *w;
	float *Tkininj, *pinj;
	trajectoryHistoryBP *history, *local_history;

	gpuErrchk(cudaMallocManaged(&w, ((blockSize * threadSize) * sizeof(double))));
	gpuErrchk(cudaMallocManaged(&Tkininj, ((blockSize * threadSize) * sizeof(float))));
	gpuErrchk(cudaMallocManaged(&pinj, ((blockSize * threadSize) * sizeof(float))));
	if (!singleTone->getInt("interactive", 0))
	{
		gpuErrchk(cudaMallocManaged(&state, ((blockSize * threadSize) * sizeof(curandState_t))));
	}
	gpuErrchk(cudaMallocHost(&local_history, ((blockSize * threadSize) * sizeof(trajectoryHistoryBP))));
	gpuErrchk(cudaMalloc(&history, ((blockSize * threadSize) * sizeof(trajectoryHistoryBP))));

	simulation.singleTone = singleTone;
	simulation.history = history;
	simulation.pinj = pinj;
	simulation.local_history = local_history;
	simulation.Tkininj = Tkininj;
	if (!singleTone->getInt("interactive", 0))
	{
		simulation.state = state;
	}
	else
	{
		simulation.state = interactiveMode->getRNGStateStructure();
	}
	simulation.w = w;
	simulation.threadSize = threadSize;
	simulation.blockSize = blockSize;

	setConstants(singleTone, true);
	runBPMethod(&simulation);

	gpuErrchk(cudaFree(w));
	gpuErrchk(cudaFree(Tkininj));
	gpuErrchk(cudaFree(pinj));
	if (!singleTone->getInt("interactive", 0))
	{
		gpuErrchk(cudaFree(state));
	}
	gpuErrchk(cudaFree(history));
	gpuErrchk(cudaFreeHost(local_history));

	AbstractAlgorithm *result;
	result = new OneDimensionBpResults();
	result->runAlgorithm(singleTone);
}

// Compute capability actual device
void OneDimensionBpAlgorithm::setThreadBlockSize()
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