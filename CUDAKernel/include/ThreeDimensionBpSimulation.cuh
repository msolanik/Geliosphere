/**
 * @file ThreeDimensionBpSimulation.cuh
 * @author Michal Solanik
 * @brief Definition of data structures needed for 3D B-p method.
 * @version 0.2
 * @date 2022-08-15
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef THREE_DIMENSION_BP_DEFINES_H
#define THREE_DIMENSION_BP_DEFINES_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#if (__CUDA_ARCH__ == 600)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#elif (__CUDA_ARCH__ == 610)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#elif (__CUDA_ARCH__ == 700)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#elif (__CUDA_ARCH__ == 750)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#elif (__CUDA_ARCH__ == 800)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#elif (__CUDA_ARCH__ == 860)
#define BLOCK_SIZE_THREE_BP 4096
#define THREAD_SIZE_THREE_BP 256
#else
#define BLOCK_SIZE_THREE_BP 64
#define THREAD_SIZE_THREE_BP 64
#endif

/**
 * @brief Data structure responsible for holding output data from 1D B-p 
 * simulations.
 * 
 */
struct trajectoryHistoryThreeDimensionBP
{
	float Tkin = -1.0f;
	float r = -1.0f;
	float theta = -1.0f;
	int id = -1;

	__device__ void setValues(float newTkin, float newR, float newTheta, int newId)
	{
		Tkin = newTkin;
		r = newR;
		theta = newTheta;
		id = newId;
	}
};

/**
 * @brief Data structure responsible for holding input information for 
 * 1D B-p method.
 * 
 */
struct simulationInputThreeDimensionBP
{
	ParamsCarrier *singleTone;
	curandState_t *state;
	trajectoryHistoryThreeDimensionBP *history;
	trajectoryHistoryThreeDimensionBP *local_history;
	float *Tkininj;
	float *pinj;
	double *w;
	int blockSize;
	int threadSize;
	int maximumSizeOfSharedMemory;
};

void runThreeDimensionBpMethod(simulationInputThreeDimensionBP *simulation);

#endif // !BP_DEFINES_H
