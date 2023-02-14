/**
 * @file OneDimensionBpSimulation.cuh
 * @author Michal Solanik
 * @brief Definition of data structures needed for 1D B-p method.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef BP_DEFINES_H
#define BP_DEFINES_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#if (__CUDA_ARCH__ == 600)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#elif (__CUDA_ARCH__ == 610)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#elif (__CUDA_ARCH__ == 700)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#elif (__CUDA_ARCH__ == 750)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#elif (__CUDA_ARCH__ == 800)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#elif (__CUDA_ARCH__ == 860)
#define BLOCK_SIZE_BP 4096
#define THREAD_SIZE_BP 256
#else
#define BLOCK_SIZE_BP 64
#define THREAD_SIZE_BP 64
#endif

/**
 * @brief Data structure responsible for holding output data from 1D B-p 
 * simulations.
 * 
 */
struct trajectoryHistoryBP
{
	float Tkin = -1.0f;
	float r = -1.0f;
	double w = -1.0f;
	int id = -1;

	__device__ void setValues(float newTkin, float newR, double newW, int newId)
	{
		Tkin = newTkin;
		r = newR;
		w = newW;
		id = newId;
	}
};

/**
 * @brief Data structure responsible for holding input information for 
 * 1D B-p method.
 * 
 */
struct simulationInputBP
{
	ParamsCarrier *singleTone;
	curandState_t *state;
	trajectoryHistoryBP *history;
	trajectoryHistoryBP *local_history;
	float *Tkininj;
	float *pinj;
	double *w;
	int blockSize;
	int threadSize;
	int maximumSizeOfSharedMemory;
};

void runBPMethod(simulationInputBP *simulation);

#endif // !BP_DEFINES_H
