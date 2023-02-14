/**
 * @file TwoDimensionBpSimulation.cu
 * @author Michal Solanik
 * @brief Implementation of 2D B-p method.
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
#include "TwoDimensionBpSimulation.cuh"
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
__global__ void wCalc(float *Tkininj, float *p, double *w, int padding)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	Tkininj[id] = getSolarPropInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float Tkinw = Tkininj[id] * 1e9f * q;
	float Rig = sqrtf(Tkinw * (Tkinw + (2.0f * T0w))) / q;
	float pinj = Rig * q / c;
	double newW = (m0_double * m0_double * c_double * c_double * c_double * c_double) + (pinj * pinj * c_double * c_double);
	newW = (pow(newW, -1.85) / pinj) / 1e45;
	w[id] = newW;
	p[id] = pinj;
}

/**
 * @brief Return value of rigidity for 2D B-p method.
 *
 * @param rigidity Current value of rigidity.
 * @return New value of rigidity.
 */
__device__ float getRigidity(float rigidity)
{
	return (rigidity < 0.1f) ? 0.1f : rigidity;
}

/**
 * @brief Run simulations for 2D B-p method.
 * This model is based on model contained in SolarProp by Niles Kappl:
 * https://arxiv.org/pdf/1511.07875.pdf
 *
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulationTwoDimensionBp(trajectoryHistoryTwoDimensionBP *history, int padding, curandState *state)
{
	extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	float r = rInit;
	float beta, Rig, dtem1, dKtt, dr, Krr, Bfactor, Kpar, dKrr, Ktt, Kper, dTkin, gamma, gamma2, dKttkon;
	float Tkin = getSolarPropInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float cosineTheta, sineTheta, dtheta, theta;
	float Bfield, Larmor, alphaH, arg, f, fprime, gamma2PlusOne2, DriftR, DriftTheta, DriftSheetR;
	theta = thetainj;
	float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_TWO_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	while (r < 100.0f)
	{
		sineTheta = sinf(theta);
		gamma = (r * omega) * sineTheta / V;
		gamma2 = gamma * gamma;

		generated[idx] = curand_box_muller(&cuState[idx]);

		// dKrr
		Rig = sqrtf(Tkin * (Tkin + (2.0f * T0)));
		dtem1 = 2.0f * r * omega * omega * sineTheta * sineTheta / (V * V);
		beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);
		dKrr = ratio * K0 * beta * Rig * ((2.0f * r * sqrtf(1.0f + gamma2)) - (r * r * dtem1 / (2.0f * sqrtf(1.0f + gamma2)))) / (1.0f + gamma2);
		dKrr = dKrr + ((1.0f - ratio) * K0 * beta * Rig * ((2.0f * r * powf(1.0f + gamma2, 1.5f)) - (r * r * dtem1 * 3.0f * sqrtf(1.0f + gamma2) / 2.0f)) / powf(1.0f + gamma2, 3.0f));
		dKrr = dKrr * 5.0f / (3.0f * 3.4f);

		Bfactor = (5.0f / 3.4f) * (r * r) / sqrtf(1.0f + gamma2);
		Kpar = K0 * beta * getRigidity(Rig) * Bfactor / 3.0f;
		// dr
		Kper = ratio * Kpar;
		Krr = Kper + ((Kpar - Kper) / (1.0f + gamma2));
		dr = ((-1.0f * V) + (2.0f * Krr / r) + dKrr) * dt;
		dr += (generated[idx].x * sqrtf(2.0f * Krr * dt));

		// dKtt
		cosineTheta = cosf(theta);
		dKtt = -1.0f * ratio * K0 * beta * Rig * r * r * sineTheta * cosineTheta;
		dKtt *= (omega * omega * r * r / (V * V));
		dKtt /= powf(1.0f + gamma2, 1.5f);
		Ktt = Kper;

		// dTheta
		dtheta = (Ktt * cosineTheta) / (r * r * sineTheta);
		dtheta = (dtheta * dt) / (1.0f + gamma2);
		dtheta = dtheta + ((generated[idx].y * sqrtf(2.0f * Ktt * dt)) / r);

		// dKttkon
		dKttkon = (Ktt * cosineTheta) / (r * r * sineTheta);
		dKttkon = dKttkon / (1.0f + gamma2);

		dTkin = -2.0f * V * ((Tkin + T0 + T0) / (Tkin + T0)) * Tkin * dt / (3.0f * r);

		// drift parameters
		Bfield = A * sqrtf((1.0f + gamma2)) / (r * r);
		Larmor = 0.0225f * Rig / Bfield;
		alphaH = Pi / sinf(alphaM + (2.0f * Larmor * Pi / (r * 180.0f)));
		alphaH = alphaH - 1.0f;
		alphaH = 1.0f / alphaH;
		alphaH = acosf(alphaH);
		arg = (1.0f - (2.0f * theta / Pi)) * tanf(alphaH);
		f = atanf(arg) / alphaH;
		fprime = 1.0f + (arg * arg);
		fprime = tanf(alphaH) / fprime;
		fprime = -1.0f * fprime * 2.0f / (Pi * alphaH);
		gamma2PlusOne2 = (1.0f + gamma2) * (1.0f + gamma2);

		// drift
		DriftR = polarity * konvF * (2.0f / (3.0f * A)) * Rig * beta * r * cosineTheta * gamma * f / ((gamma2PlusOne2)*sineTheta);
		DriftSheetR = polarity * konvF * (1.0f / (3.0f * A)) * Rig * beta * r * gamma * fprime / (1.0f + gamma2);
		dr = dr + ((DriftR + DriftSheetR) * dt);
		DriftTheta = driftThetaConstant * Rig * beta * r * gamma * (2.0f + (gamma2)) * f / (gamma2PlusOne2);
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
 * @brief Run 2D B-p method with given parameters defines
 * in input simulation data structure.
 *
 */
void runTwoDimensionBpMethod(simulationInputTwoDimensionBP *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
	spdlog::info("Starting initialization of 2D B-p simulation.");

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
		spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("2DBP", destination))
	{
		spdlog::error("Directory for 2D B-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) * 1000000 / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->maximumSizeOfSharedMemory != -1)
	{
		cudaFuncSetAttribute(trajectorySimulationTwoDimensionBp, cudaFuncAttributeMaxDynamicSharedMemorySize, simulation->maximumSizeOfSharedMemory);
	}
	for (int k = 0; k < iterations; ++k)
	{
		spdlog::info("Processed: {:03.2f}%", (float)k / ((float)iterations / 100.0));
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		wCalc<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, simulation->w, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulationTwoDimensionBp<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
		spdlog::info("In this iteration {} particles were detected.", counter);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryTwoDimensionBP), cudaMemcpyDeviceToHost));
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