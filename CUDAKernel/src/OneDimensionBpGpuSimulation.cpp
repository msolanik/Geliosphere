#include "OneDimensionBpGpuSimulation.hpp"
#include "CudaErrorCheck.cuh"
#include "CosmicUtils.cuh"
#include "CosmicConstants.cuh"
#include "OneDimensionBpSimulation.cuh"

void OneDimensionBpGpuSimulation::prepareAndRunSimulation(ParamsCarrier *singleTone)
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
    gpuErrchk(cudaMallocManaged(&state, ((blockSize * threadSize) * sizeof(curandState_t))));
    gpuErrchk(cudaMallocHost(&local_history, ((blockSize * threadSize) * sizeof(trajectoryHistoryBP))));
    gpuErrchk(cudaMalloc(&history, ((blockSize * threadSize) * sizeof(trajectoryHistoryBP))));

    simulation.singleTone = singleTone;
    simulation.history = history;
    simulation.pinj = pinj;
    simulation.local_history = local_history;
    simulation.Tkininj = Tkininj;
    simulation.state = state;
    simulation.w = w;
    simulation.threadSize = threadSize;
    simulation.blockSize = blockSize;
    simulation.maximumSizeOfSharedMemory = sharedMemoryMaximumSize;

    setConstants(singleTone);
    runBPMethod(&simulation);

    gpuErrchk(cudaFree(w));
    gpuErrchk(cudaFree(Tkininj));
    gpuErrchk(cudaFree(pinj));
    gpuErrchk(cudaFree(state));
    gpuErrchk(cudaFree(history));
    gpuErrchk(cudaFreeHost(local_history));
}

// Compute capability actual device
void OneDimensionBpGpuSimulation::setThreadBlockSize()
{
	cudaDeviceProp gpuProperties;
	gpuErrchk(cudaGetDeviceProperties(&gpuProperties, 0));
	int computeCapability = gpuProperties.major * 100 + gpuProperties.minor * 10;
    switch (computeCapability)
    {
    case 600:
    case 610:
    case 700:
    case 800:
    case 860:
    case 750:
        blockSize = 4096;
        threadSize = 256;
        break;
    default:
        blockSize = 64;
        threadSize = 64;
        break;
    }

    switch (computeCapability)
    {
    case 600:
        sharedMemoryMaximumSize = 65536;
        break;
    case 610:
        sharedMemoryMaximumSize = 98304;
        break;
    case 700:
        sharedMemoryMaximumSize = 98304;
        break;
    case 800:
        sharedMemoryMaximumSize = 167936;
        break;
    case 860:
        sharedMemoryMaximumSize = 102400;
        break;
    case 750:
        sharedMemoryMaximumSize = 65536;
        break;
    default:
        sharedMemoryMaximumSize = -1;
        break;
    }
}