/**
 * @file OneDimensionFpSimulation.cuh
 * @author Michal Solanik
 * @brief Definition of data structures needed for 1D F-p method.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef FLOAT_FW_DEFINES_H
#define FLOAT_FW_DEFINES_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#if (__CUDA_ARCH__ == 610)
#define BLOCK_SIZE 65536
#define THREAD_SIZE 512
#elif (__CUDA_ARCH__ == 750)
#define BLOCK_SIZE 32768
#define THREAD_SIZE 1024
#else
#define BLOCK_SIZE 64
#define THREAD_SIZE 64
#endif

/**
 * @brief Data structure responsible for holding output data from 1D F-p 
 * simulations.
 * 
 */
struct trajectoryHistory
{
	float sumac = -1.0f;
	float r = -1.0f;
	float p = -1.0f;
	int id = -1;

	__device__ void setValues(float newSumac, float newR, float newP, int newId)
	{
		sumac = newSumac;
		r = newR;
		p = newP;
		id = newId;
	}
};

/**
 * @brief Data structure responsible for holding input information for 
 * 1D F-p method.
 * 
 */
struct simulationInput
{
	ParamsCarrier *singleTone;
	curandState_t *state;
	trajectoryHistory *history;
	trajectoryHistory *local_history;
	float *pinj;
	double *w;
	int blockSize;
	int threadSize;
};

void runFWMethod(simulationInput *simulation);

#endif // !FLOAT_FW_DEFINES_H
