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

__device__ float getSolarPropInjection(unsigned long long state)
{
	int modulo = state % 30;
	return 0.01f * powf((1.0f + 0.5f), modulo); 
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