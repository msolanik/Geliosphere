/**
 * @file OneDimensionBpSimulation.cu
 * @author Michal Solanik
 * @brief Implementation of 1D B-p method.
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>
#include <string>

#include "ParamsCarrier.hpp"
#include "FileUtils.hpp"
#include "OneDimensionBpSimulation.cuh"
#include "CosmicConstants.cuh"
#include "CosmicUtils.cuh"
#include "CudaErrorCheck.cuh"

extern "C" void runBPMethod(simulationInputBP *simulation);

/**
 * @brief Calculate pre-simulations parameters.
 * 
 * @param Tkininj Injecting kinetic energy.
 * @param pinj Injecting particle momentum.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 */
__global__ void wCalcBP(float *Tkininj, float *pinj, int padding)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float Tkin = getTkinInjection(BLOCK_SIZE_BP * THREAD_SIZE_BP * padding + id);
	float Rig = sqrtf(Tkin * (Tkin + (2 * T0)));
	float p = Rig * 1e9 * q / c;
	pinj[id] = p;
	Tkininj[id] = Tkin;
}

/**
 * @brief Run simulations for 1D B-p method. 
 * More information about approach choosed for 1D B-p model can be found here:
 * https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237
 * 
 * @param pinj Injecting particle momentum.
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulationBP(float *pinj, trajectoryHistoryBP *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = 1.0f;
	float p = pinj[id];
	float beta, Rig, dr, pp;
	float Tkin = getTkinInjection(BLOCK_SIZE_BP * THREAD_SIZE_BP * padding + id);
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	bool generate = true;
	for (; r < 100.0002f;)
	{
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		Rig = (p * c / q) / 1e9;
		pp = p;
		p -= (2.0f * V * pp * dt / (3.0f * r));
		if (generate)
		{
			generated[idx] = curand_box_muller(&cuState[idx]);
			dr = (V + (2.0f * K0 * beta * Rig / r)) * dt + (generated[idx].x * sqrtf(2.0f * K0 * beta * Rig * dt));
			r += dr;
			generate = false;
		}
		else
		{
			dr = (V + (2.0f * K0 * beta * Rig / r)) * dt + (generated[idx].y * sqrtf(2.0f * K0 * beta * Rig * dt));
			r += dr;
			generate = true;
		}
		Rig = p * c / q;
		Tkin = (sqrtf((T0 * T0 * q * q * 1e9f * 1e9f) + (q * q * Rig * Rig)) - (T0 * q * 1e9f)) / (q * 1e9f);
		Rig = Rig / 1e9;
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		if (beta > 0.01f && Tkin < 200.0f)
		{
			if ((r > 100.0f) && ((r - dr) < 100.0f))
			{
				count = atomicAdd(&outputCounter, 1);
				double newW = (m0_double * m0_double * c_double * c_double * c_double * c_double) + (p * p * c_double * c_double);
				newW = (pow(newW, -1.85) / p) / 1e45;
				history[count].setValues(Tkin, r, newW, id);
				break;
			}
		}
		else if (beta < 0.01f)
		{
			break;
		}
		if (r < 0.3f)
		{
			r -= dr;
			p = pp;
		}
	}
	state[id] = cuState[idx];
}

/**
 * @brief Run 1D B-p method with given parameters defines 
 * in input simulation data structure.
 * 
 */
void runBPMethod(simulationInputBP *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
	}
	if (!createDirectory("BP", destination))
	{
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = singleTone->getInt("bilions", 1);
	if (simulation->threadSize == 1024)
	{
		cudaFuncSetAttribute(trajectorySimulationBP, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
	}
	for (int k = 0; k < iterations; ++k)
	{
		for (int i = 0; i < 31; ++i)
		{
			nullCount<<<1, 1>>>();
			gpuErrchk(cudaDeviceSynchronize());
			wCalcBP<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, i);
			gpuErrchk(cudaDeviceSynchronize());
			trajectorySimulationBP<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->pinj, simulation->history, i, simulation->state);
			gpuErrchk(cudaDeviceSynchronize());
			cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
			if (counter != 0)
			{
				gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryBP), cudaMemcpyDeviceToHost));
				for (int j = 0; j < counter; ++j)
				{
					fprintf(file, "%g %g %g %g\n", simulation->local_history[j].Tkin, simulation->Tkininj[simulation->local_history[j].id],
							simulation->local_history[j].r, simulation->local_history[j].w);
				}
			}
		}
	}
	fclose(file);
}