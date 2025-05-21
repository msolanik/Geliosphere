/**
 * @file SolarPropLikeModel.cu
 * @author Michal Solanik
 * @brief Implementation of SolarPropLike model.
 * @version 0.2
 * @date 2022-06-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>
#include <string>

#include "spdlog/spdlog.h"

#include "ParamsCarrier.hpp"
#include "FileUtils.hpp"
#include "SolarPropLikeModel.cuh"
#include "CosmicConstants.cuh"
#include "CosmicUtils.cuh"
#include "CudaErrorCheck.cuh"

/**
 * @brief Calculate pre-simulations parameters.
 *
 * @param Tkininj Injecting kinetic energy.
 * @param p Particle momentum.
 * @param w Spectrum intensity.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 */
__global__ void trajectoryPreSimulation(float *Tkininj, float *p, double *w, int padding)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	Tkininj[id] = (useUniformInjection) 
        ? getTkinInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id)
        : getSolarPropInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float Tkinw = Tkininj[id] * 1e9f * q;
	
	// Equation 8 from 
    // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
    // https://arxiv.org/pdf/1511.07875.pdf		
	float Rig = sqrtf(Tkinw * (Tkinw + (2.0f * T0w))) / q;
	float pinj = Rig * q / c;

	// Equation under Equation 6 from 
    // Yamada et al. "A stochastic view of the solar modulation phenomena of cosmic rays" GEOPHYSICAL RESEARCH LETTERS, VOL. 25, NO.13, PAGES 2353-2356, JULY 1, 1998
    // https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL51869
	double newW = (m0_double * m0_double * c_double * c_double * c_double * c_double) + (pinj * pinj * c_double * c_double);
	newW = (pow(newW, -1.85) / pinj) / 1e45;
	w[id] = newW;
	p[id] = pinj;
}

/**
 * @brief Return value of rigidity for SolarPropLike model.
 *
 * @param rigidity Current value of rigidity.
 * @return New value of rigidity.
 */
__device__ float getRigidity(float rigidity)
{
	return (rigidity < 0.1f) ? 0.1f : rigidity;
}

/**
 * @brief Run simulations for SolarPropLike model.
 * This model is based on model contained in SolarProp by Niles Kappl:
 * https://arxiv.org/pdf/1511.07875.pdf
 *
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulationSolarPropLike(trajectoryHistorySolarPropLike *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = rInit;
	float beta, Rig, dKrrTmp, dr, Krr, Bfactor, Kpar, dKrr, Ktt, Kper, dTkin, gamma, gamma2;
	float Tkin = (useUniformInjection) 
        ? getTkinInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id)
        : getSolarPropInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float cosineTheta, sineTheta, dtheta, theta;
	float Bfield, Larmor, alphaH, arg, f, fprime, gamma2PlusOne2, DriftR, DriftTheta, DriftSheetR;
	theta = thetainj;
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_TWO_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	while (r < 100.0f)
	{
		// Support variables
		sineTheta = sinf(theta);
		gamma = (r * omega) * sineTheta / V;
		gamma2 = gamma * gamma;

		generated[idx] = curand_box_muller(&cuState[idx]);

        // Equation 8 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf			
		Rig = sqrtf(Tkin * (Tkin + (2.0f * T0)));
		
		// Equation 5
        // Link to Equation 5 in Jupyter Notebook Documentation: 
        // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#5
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		
		// Equation 44 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dKrrTmp = 2.0f * r * omega * omega * sineTheta * sineTheta / (V * V);
		dKrr = ratio * K0 * beta * Rig * ((2.0f * r * sqrtf(1.0f + gamma2)) - (r * r * dKrrTmp / (2.0f * sqrtf(1.0f + gamma2)))) / (1.0f + gamma2);
		dKrr = dKrr + ((1.0f - ratio) * K0 * beta * Rig * ((2.0f * r * powf(1.0f + gamma2, 1.5f)) - (r * r * dKrrTmp * 3.0f * sqrtf(1.0f + gamma2) / 2.0f)) / powf(1.0f + gamma2, 3.0f));
		dKrr = dKrr * 5.0f / (3.0f * 3.4f);

		Bfactor = (5.0f / 3.4f) * (r * r) / sqrtf(1.0f + gamma2);
		
		// Equation 42 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		Kpar = K0 * beta * getRigidity(Rig) * Bfactor / 3.0f;
		
		// Equation 43 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		Kper = ratio * Kpar;
		
		// Equation 14 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		Krr = Kper + ((Kpar - Kper) / (1.0f + gamma2));

		// Equation 16 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dr = ((-1.0f * V) + (2.0f * Krr / r) + dKrr) * dt;
		dr += (generated[idx].x * sqrtf(2.0f * Krr * dt));

		cosineTheta = cosf(theta);

		// Equation 15 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		Ktt = Kper;

		// Equation 17 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dtheta = (Ktt * cosineTheta) / (r * r * sineTheta);
		dtheta = (dtheta * dt) / (1.0f + gamma2);
		dtheta = dtheta + ((generated[idx].y * sqrtf(2.0f * Ktt * dt)) / r);

		// Equation 18 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dTkin = -2.0f * V * ((Tkin + T0 + T0) / (Tkin + T0)) * Tkin * dt / (3.0f * r);

		Bfield = A * sqrtf((1.0f + gamma2)) / (r * r);
		
		// Equation 26 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		Larmor = 0.0225f * Rig / Bfield;

		// Equation 34 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		alphaH = Pi / sinf(alphaM + (2.0f * Larmor * Pi / (r * 180.0f)));
		alphaH = alphaH - 1.0f;
		alphaH = 1.0f / alphaH;
		alphaH = acosf(alphaH);

		// Equation 32 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		arg = (1.0f - (2.0f * theta / Pi)) * tanf(alphaH);
		f = atanf(arg) / alphaH;

		// Equation 33 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		fprime = 1.0f + (arg * arg);
		fprime = tanf(alphaH) / fprime;
		fprime = -1.0f * fprime * 2.0f / (Pi * alphaH);
		
		// Support variables
		gamma2PlusOne2 = (1.0f + gamma2) * (1.0f + gamma2);

    	// Equation 35 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		DriftR = polarity * konvF * (2.0f / (3.0f * A)) * Rig * beta * r * cosineTheta * gamma * f / ((gamma2PlusOne2)*sineTheta);

		// Equation 37 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		DriftSheetR = polarity * konvF * (1.0f / (3.0f * A)) * Rig * beta * r * gamma * fprime / (1.0f + gamma2);

		// Equation 16 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dr = dr + ((DriftR + DriftSheetR) * dt);

		// Equation 36 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		DriftTheta = driftThetaConstant * Rig * beta * r * gamma * (2.0f + (gamma2)) * f / (gamma2PlusOne2);

		// Equation 17 from 
        // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
        // https://arxiv.org/pdf/1511.07875.pdf
		dtheta += (DriftTheta * dt / r);

		r = r + dr;
		theta = theta + dtheta;
		Tkin = Tkin - dTkin;

		if (theta < 0.0f)
		{
			theta = fabsf(theta);
		}
		if (theta > 2.0f * Pi)
		{
			theta = theta - (2.0f * Pi);
		}
		else if (theta > Pi)
		{
			theta = (2.0f * Pi) - theta;
		}

		if (r < 0.0f)
		{
			r = -1.0f * r;
		}
		if (r > 100.0f)
		{
			count = atomicAdd(&outputCounter, 1);
			history[count].setValues(Tkin, r, theta, id);
			break;
		}
	}
	state[id] = cuState[idx];
}

/**
 * @brief Run SolarPropLike model with given parameters defines
 * in input simulation data structure.
 *
 */
void runSolarPropLikeSimulation(simulationInputSolarPropLike *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
	spdlog::info("Starting initialization of SOLARPROPlike 2D simulation.");

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
		spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("SOLARPROPLike", destination))
	{
		spdlog::error("Directory for SOLARPROPlike 2D simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) * 1000000 / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->maximumSizeOfSharedMemory != -1)
	{
		cudaFuncSetAttribute(trajectorySimulationSolarPropLike, cudaFuncAttributeMaxDynamicSharedMemorySize, simulation->maximumSizeOfSharedMemory);
	}
	for (int k = 0; k < iterations; ++k)
	{
		spdlog::info("Processed: {:03.2f}%", (float)k / ((float)iterations / 100.0));
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		trajectoryPreSimulation<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, simulation->w, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulationSolarPropLike<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
		spdlog::info("In this iteration {} particles were detected.", counter);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistorySolarPropLike), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, "%g %g %g %g %g %g\n", simulation->Tkininj[simulation->local_history[j].id], simulation->local_history[j].Tkin, simulation->local_history[j].r, simulation->w[simulation->local_history[j].id], singleTone->getFloat("theta_injection", 90.0f) * 3.1415926535f / 180.0f, simulation->local_history[j].theta);
			}
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
	spdlog::info("Simulation ended.");
}