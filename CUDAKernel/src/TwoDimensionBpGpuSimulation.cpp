#include "TwoDimensionBpGpuSimulation.hpp"
#include "CudaErrorCheck.cuh"
#include "CosmicUtils.cuh"
#include "CosmicConstants.cuh"
#include "TwoDimensionBpSimulation.cuh"

void TwoDimensionBpGpuSimulation::prepareAndRunSimulation(ParamsCarrier *singleTone)
{
    simulationInputTwoDimensionBP simulation;
    setThreadBlockSize();
    curandState_t *state;
    double *w;
    float *pinj;
    float *Tkininj;
    trajectoryHistoryTwoDimensionBP *history, *local_history;

    gpuErrchk(cudaMallocManaged(&w, ((blockSize * threadSize) * sizeof(double))));
    gpuErrchk(cudaMallocManaged(&pinj, ((blockSize * threadSize) * sizeof(float))));
    gpuErrchk(cudaMallocManaged(&Tkininj, ((blockSize * threadSize) * sizeof(float))));
    gpuErrchk(cudaMallocManaged(&state, ((blockSize * threadSize) * sizeof(curandState_t))));
    gpuErrchk(cudaMallocHost(&local_history, ((blockSize * threadSize) * sizeof(trajectoryHistoryTwoDimensionBP))));
    gpuErrchk(cudaMalloc(&history, ((blockSize * threadSize) * sizeof(trajectoryHistoryTwoDimensionBP))));

    simulation.singleTone = singleTone;
    simulation.history = history;
    simulation.local_history = local_history;
    simulation.pinj = pinj;
    simulation.Tkininj = Tkininj;
    simulation.state = state;
    simulation.w = w;
    simulation.threadSize = threadSize;
    simulation.blockSize = blockSize;
    singleTone->putFloat("K0", 1.43831e-5f);

    setConstants(singleTone);
    runTwoDimensionBpMethod(&simulation);

    gpuErrchk(cudaFree(w));
    gpuErrchk(cudaFree(pinj));
    gpuErrchk(cudaFree(Tkininj));
    gpuErrchk(cudaFree(state));
    gpuErrchk(cudaFree(history));
    gpuErrchk(cudaFreeHost(local_history));
}

// Compute capability actual device
void TwoDimensionBpGpuSimulation::setThreadBlockSize()
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