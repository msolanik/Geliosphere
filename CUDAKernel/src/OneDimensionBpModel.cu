/**
 * @file OneDimensionBpModel.cu
 * @author Michal Solanik
 * @brief Implementation of 1D B-p model.
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>
#include <math.h>
#include <string>

#include "spdlog/spdlog.h"

#include "ParamsCarrier.hpp"
#include "FileUtils.hpp"
#include "OneDimensionBpModel.cuh"
#include "CosmicConstants.cuh"
#include "CosmicUtils.cuh"
#include "CudaErrorCheck.cuh"

/**
 * @brief Calculate pre-simulations parameters.
 * 
 * @param Tkininj Injecting kinetic energy.
 * @param pinj Injecting particle momentum.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 */
__global__ void trajectoryPreSimulationBP(float *Tkininj, float *pinj, int padding)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float Tkin = getTkinInjection(BLOCK_SIZE_BP * THREAD_SIZE_BP * padding + id);
	float Rig = sqrtf(Tkin * (Tkin + (2.0 * T0)));
	float p = Rig * 1e9 * q / c;
	pinj[id] = p;
	Tkininj[id] = Tkin;
}

/**
 * @brief Run simulations for 1D B-p model. 
 * More information about approach choosed for 1D B-p model can be found here:
 * https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237
 * 
 * @param pinj Injecting particle momentum.
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulationBP(float *pinj, trajectoryHistoryOneDimensionBp *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = rInit;
	float p = pinj[id];
	float beta, Rig, dr, pp;
	float Tkin = getTkinInjection(BLOCK_SIZE_BP * THREAD_SIZE_BP * padding + id);
	float Kdiff;
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	bool generate = true;
	while (r < 100.0002f)
	{
		// Equation 5
        // Link to Equation 5 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#5
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		
		// Equation 6 in GeV
        // Link to Equation 6 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#6
		Rig = (p * c / q) / 1e9f;
		pp = p;

		// Equation 14
        // Link to Equation 14 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#14
		p -= (2.0f * V * pp * dt / (3.0f * r));

		// Equation 7
        // Link to Equation 7 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#7
		Kdiff = K0 * beta * Rig;
		if (generate)
		{
			generated[idx] = curand_box_muller(&cuState[idx]);
			// Equation 13
			// Link to Equation 13 in Jupyter Notebook Documentation: 
			// https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#13
			dr = (V + (2.0f * Kdiff / r)) * dt + (generated[idx].x * sqrtf(2.0f * Kdiff * dt));
			r += dr;
			generate = false;
		}
		else
		{
			// Equation 13
			// Link to Equation 13 in Jupyter Notebook Documentation: 
			// https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#13
			dr = (V + (2.0f * Kdiff / r)) * dt + (generated[idx].y * sqrtf(2.0f * Kdiff * dt));
			r += dr;
			generate = true;
		}
		// Equation 6 in J
        // Link to Equation 6 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#6
		Rig = p * c / q;
		Tkin = (sqrtf((T0 * T0 * q * q * 1e9f * 1e9f) + (q * q * Rig * Rig)) - (T0 * q * 1e9f)) / (q * 1e9f);
		
		// Equation 5
        // Link to Equation 5 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#5
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
 * @brief Run 1D B-p model with given parameters defines 
 * in input simulation data structure.
 * 
 */
void runOneDimensionBpSimulation(simulationInputBP *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
    spdlog::info("Starting initialization of 1D B-p simulation.");
	
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
        spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("1DBP", destination))
	{
        spdlog::error("Directory for 1D B-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) * 1000000 / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->maximumSizeOfSharedMemory != -1)
	{
		cudaFuncSetAttribute(trajectorySimulationBP, cudaFuncAttributeMaxDynamicSharedMemorySize, simulation->maximumSizeOfSharedMemory);
	}
	for (int k = 0; k < iterations; ++k)
	{
        spdlog::info("Processed: {:03.2f}%", (float)k / ((float)iterations / 100.0));
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		trajectoryPreSimulationBP<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulationBP<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->pinj, simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        spdlog::info("In this iteration {} particles were detected.", counter);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryOneDimensionBp), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, "%g %g %g %g\n", simulation->local_history[j].Tkin, simulation->Tkininj[simulation->local_history[j].id],
						simulation->local_history[j].r, simulation->local_history[j].w);
			}
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
    spdlog::info("Simulation ended.");
}