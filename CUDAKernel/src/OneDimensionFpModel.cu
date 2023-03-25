/**
 * @file OneDimensionFpModel.cu
 * @author Michal Solanik
 * @brief Implementation of 1D F-p model.
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>
#include <string>

#include "spdlog/spdlog.h"

#include "ParamsCarrier.hpp"
#include "FileUtils.hpp"
#include "OneDimensionFpModel.cuh"
#include "CosmicConstants.cuh"
#include "CudaErrorCheck.cuh"
#include "CosmicUtils.cuh"

/**
 * @brief Calculate pre-simulations parameters.
 * 
 * @param w Spectrum intensity.
 * @param pinj Injecting particle momentum.
 * @param iteration Support value used to calculate state for getting
 * kinetic energy.
 */
__global__ void trajectoryPreSimulation(double *w, float *pinj, int iteration)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float Tkinw = getTkinInjection(BLOCK_SIZE * THREAD_SIZE * iteration + id) * 1e9f * q;
	float Rig = sqrtf(Tkinw * (Tkinw + (2.0f * T0w))) / q;
	float p = Rig * q / c;
	
	// Equation under Equation 6 from 
    // Yamada et al. "A stochastic view of the solar modulation phenomena of cosmic rays" GEOPHYSICAL RESEARCH LETTERS, VOL. 25, NO.13, PAGES 2353-2356, JULY 1, 1998
    // https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL51869
	double newW = (m0_double * m0_double * c_double * c_double * c_double * c_double) + (p * p * c_double * c_double);
	newW = (pow(newW, -1.85) / p) / 1e45;
	w[id] = newW;
	pinj[id] = p;
}

/**
 * @brief Run simulations for 1D F-p model. 
 * More information about approach choosed for 1D B-p model can be found here:
 * https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237
 * 
 * @param pinj Injecting particle momentum.
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulation(float *pinj, trajectoryHistoryOneDimensionFp *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = 100.0001f;
	float p = pinj[id];
	float beta, sumac = 0.0f, Rig, dr, pp, Kdiff;
	float Tkin = getTkinInjection(BLOCK_SIZE * THREAD_SIZE * padding + id);
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE]);
	cuState[idx] = state[id];
	int count;
	bool generate = true;
	while (r < 100.0002f)
	{
		// Equation 5
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		
		// Equation 6
		Rig = sqrtf(Tkin * (Tkin + (2.0f * T0)));
		
		// Equation 11
		sumac += ((4.0f * V / (3.0f * r)) * dt);
		
		// Equation 10
		pp = p;
		p -= (2.0f * V * pp * dt / (3.0f * r));
	
		// Equation 7
		Kdiff = K0 * beta * Rig;
		if (generate)
		{
			generated[idx] = curand_box_muller(&cuState[idx]);
			// Equation 9
			dr = (V + (2.0f * Kdiff / r)) * dt + (generated[idx].x * sqrtf(2.0f * Kdiff * dt));
			r += dr;
			generate = false;
		}
		else
		{
			// Equation 9
			dr = (V + (2.0f * Kdiff / r)) * dt + (generated[idx].y * sqrtf(2.0f * Kdiff * dt));
			r += dr;
			generate = true;
		}
		// Equation 6 in J
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
 * @brief Run 1D F-p model with given parameters defines 
 * in input simulation data structure.
 * 
 */
void runOneDimensionFpSimulation(simulationInput *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
    spdlog::info("Starting initialization of 1D F-p simulation.");

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
        spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("1DFP", destination))
	{
        spdlog::error("Directory for 1D F-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) * 1000000  / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->maximumSizeOfSharedMemory != -1)
	{
		cudaFuncSetAttribute(trajectorySimulation, cudaFuncAttributeMaxDynamicSharedMemorySize, simulation->maximumSizeOfSharedMemory);
	}
	for (int iteration = 0; iteration < iterations; ++iteration)
	{
        spdlog::info("Processed: {:03.2f}%", (float)iteration / ((float)iterations / 100.0));
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		trajectoryPreSimulation<<<simulation->blockSize, simulation->threadSize>>>(simulation->w, simulation->pinj, iteration);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulation<<<simulation->blockSize, simulation->threadSize, 
			simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->pinj, 
				simulation->history, iteration, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        spdlog::info("In this iteration {} particles were detected.", counter);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryOneDimensionFp), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, " %g  %g  %g  %g %g \n", simulation->pinj[simulation->local_history[j].id], simulation->local_history[j].p,
						simulation->local_history[j].r, simulation->w[simulation->local_history[j].id], simulation->local_history[j].sumac);
			}
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
    spdlog::info("Simulation ended.");
}