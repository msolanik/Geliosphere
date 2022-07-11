/**
 * @file OneDimensionFpSimulation.cu
 * @author Michal Solanik
 * @brief Implementation of 1D F-p method.
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
#include "OneDimensionFpSimulation.cuh"
#include "CosmicConstants.cuh"
#include "CudaErrorCheck.cuh"
#include "CosmicUtils.cuh"

/**
 * @brief Calculate pre-simulations parameters.
 * 
 * @param w Spectrum intensity.
 * @param pinj Injecting particle momentum.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 */
__global__ void wCalc(double *w, float *pinj, int padding)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float Tkinw = getTkinInjection(BLOCK_SIZE * THREAD_SIZE * padding + id) * 1e9f * q;
	float Rig = sqrtf(Tkinw * (Tkinw + (2 * T0w))) / q;
	float p = Rig * q / c;
	double newW = (m0_double * m0_double * c_double * c_double * c_double * c_double) + (p * p * c_double * c_double);
	newW = (pow(newW, -1.85) / p) / 1e45;
	w[id] = newW;
	pinj[id] = p;
}

/**
 * @brief Run simulations for 1D F-p method. 
 * More information about approach choosed for 1D B-p model can be found here:
 * https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237
 * 
 * @param pinj Injecting particle momentum.
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulation(float *pinj, trajectoryHistory *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = 100.0001f;
	float p = pinj[id];
	float beta, sumac = 0.0f, Rig, dr, pp;
	float Tkin = getTkinInjection(BLOCK_SIZE * THREAD_SIZE * padding + id);
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE]);
	cuState[idx] = state[id];
	int count;
	bool generate = true;
	for (; r < 100.0002f;)
	{
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		Rig = sqrtf(Tkin * (Tkin + (2.0f * T0)));
		sumac += ((4.0f * V / (3.0f * r)) * dt);
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
		if (beta > 0.01f && Tkin < 100.0f)
		{
			if ((r - 1.0f) / ((r - dr) - 1.0f) < 0.0f)
			{
				count = atomicAdd(&outputCounter, 1);
				history[count].setValues(sumac, r, p, id);
			}
		}
		else if (beta < 0.01f)
		{
			break;
		}
		if (r < 0.1f)
		{
			r -= dr;
			p = pp;
		}
	}
	state[id] = cuState[idx];
}

/**
 * @brief Run 1D F-p method with given parameters defines 
 * in input simulation data structure.
 * 
 */
void runFWMethod(simulationInput *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
	}
	if (!createDirectory("FW", destination))
	{
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->threadSize == 1024)
	{
		cudaFuncSetAttribute(trajectorySimulation, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
	}
	for (int k = 0; k < iterations; ++k)
	{
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		wCalc<<<simulation->blockSize, simulation->threadSize>>>(simulation->w, simulation->pinj, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulation<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->pinj, simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistory), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, " %g  %g  %g  %g %g \n", simulation->pinj[simulation->local_history[j].id], simulation->local_history[j].p,
						simulation->local_history[j].r, simulation->w[simulation->local_history[j].id], simulation->local_history[j].sumac);
			}
		}
	}
	fclose(file);
}