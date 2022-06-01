/**
 * @file CosmicUtils.cuh
 * @author Michal Solanik
 * @brief Implementation of common functions for simulations.
 * @version 0.1
 * @date 2021-07-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "CosmicUtils.cuh"
#include "CosmicConstants.cuh"

extern "C" void initRNG(curandState_t *state, int blockSize, int threadSize);

__device__ int outputCounter = 0;

__device__ float getTkinInjection(unsigned long long state)
{
	unsigned long long ownState = state;
	int modulo;
	if (ownState > (injectionMax * quantityPerEnergy))
	{
		ownState -= (__double2ll_rd(ownState / (injectionMax * quantityPerEnergy)) * (injectionMax * quantityPerEnergy));
	}
	if (ownState >= quantityPerEnergy)
	{
		modulo = __double2int_rd(ownState / quantityPerEnergy);
	}
	else
	{
		modulo = 0;
	}
	return ((modulo) + ((ownState - (modulo * quantityPerEnergy) + 1) / quantityPerEnergy));
}

__global__ void nullCount()
{
	outputCounter = 0;
}

__global__ void curandInitialization(curandState_t *state)
{
	int execID = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock(), execID, 0, &state[execID]);
}

void initRNG(curandState_t *state, int blockSize, int threadSize)
{
	curandInitialization<<<blockSize, threadSize>>>(state);
	gpuErrchk(cudaDeviceSynchronize());
}