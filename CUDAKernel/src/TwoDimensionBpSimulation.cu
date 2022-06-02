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

#include "ParamsCarrier.hpp"
#include "FileUtils.hpp"
#include "TwoDimensionBpSimulation.cuh"
#include "CosmicConstants.cuh"
#include "CosmicUtils.cuh"
#include "CudaErrorCheck.cuh"

extern "C" void runTwoDimensionBpMethod(simulationInputTwoDimensionBP *simulation);

__global__ void wCalc(float *Tkininj, float *p, double *w, int padding) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x; 
    Tkininj[id] = getTkinInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float Tkinw = Tkininj[id]*1e9f*q;
	float Rig = sqrtf(Tkinw*(Tkinw + (2.0f * T0w))) / q;
	float pinj = Rig*q / c;
	double newW = (m0_double*m0_double*c_double*c_double*c_double*c_double) + (pinj*pinj*c_double*c_double);
	newW = (pow(newW, -1.85) / pinj) / 1e45;
	w[id] = newW;
	p[id] = pinj;
}

__device__ float getRigidity(float rigidity) {
	return (rigidity < 0.1f) ? 0.1f : rigidity;
}

__global__ void trajectorySimulationTwoDimensionBp(float *pinj, trajectoryHistoryTwoDimensionBP* history, int padding, curandState* state) {
    extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x; 
	int idx = threadIdx.x;
	float r =  1.0f;
    float p = pinj[id]; 
    float beta, Rig, dtem1, dKtt, dr, Krr, Bfactor, Kpar, dKrr, Ktt, Kper, dTkin, gamma, gamma2, dKttkon;
	float Tkin = getTkinInjection(BLOCK_SIZE_TWO_BP * THREAD_SIZE_TWO_BP * padding + id);
	float cosineTheta, sineTheta, previousTheta, dtheta, theta, gammaPlusOne;
	float Bfield, Larmor, alphaH, arg, f, fprime, gamma2PlusOne2, DriftR, DriftTheta, DriftSheetR;
	theta = thetainj;
    float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_TWO_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	for (; r < 100.0002f;) {
		sineTheta = sinf(theta);
		gamma = (r * omega) * sineTheta / V;
		gamma2 = gamma * gamma;

		generated[idx] = curand_box_muller(&cuState[idx]);

		// dKrr
		Rig = sqrtf(Tkin*(Tkin + (2.0f * T0)));
		dtem1 = 2.0f*r*omega*omega*sineTheta*sineTheta/(V*V);
		gammaPlusOne = 1.0f + gamma2;
		beta = sqrtf(Tkin*(Tkin + T0 + T0)) / (Tkin + T0);
		dKrr = ratio*K0*beta*Rig;
		dKrr *= ((2.0f*r*sqrtf(gammaPlusOne)) - (r*r*dtem1/(2.0f*sqrtf(gammaPlusOne))));
		dKrr /= (gammaPlusOne); 
		dKrr += ((1.0f-ratio)*K0*beta*Rig*((2.0f*r*powf((gammaPlusOne),1.5f)) - (r*r*dtem1*3.0f*sqrtf(gammaPlusOne)/2.0f))/powf((gammaPlusOne),3.0f));
		dKrr = dKrr*5.0f/(3.0f*3.4f);     // SOLARPROP


		Bfactor = (5.0f/3.4f) *  (r * r) / sqrt(1.0f + gamma2);
        Kpar = K0*beta*getRigidity(Rig)*Bfactor/3.0f;
		// dr
		Kper = ratio * Kpar;   // SOLARPROP
		Krr = Kper + ((Kpar - Kper)/(1.0f + gamma2));
		dr = ((-1.0f*V) + (2.0f*Krr/r) + dKrr)*dt; 
		dr += (generated[idx].x*sqrtf(2.0f*Krr*dt));

		// dKtt
		cosineTheta = cosf(theta);
		dKtt = -1.0f*ratio*K0*beta*Rig*r*r*sineTheta*cosineTheta;
		dKtt *= (omega*omega*r*r/(V*V));
		dKtt /= powf(1.0f + gamma2, 1.5f);
		Ktt = Kper;

		// dTheta
		dtheta = (Ktt*cosineTheta) / (r * r * sineTheta);  
      	dtheta = (dtheta*dt) + (dKtt*dt/(r *r));
      	dtheta = dtheta +  ((generated[idx].y*sqrt(2.0f*Ktt*dt)) / r); 

		// dKttkon
		dKttkon = (Ktt*cosineTheta) / (r * r * sineTheta);
		dKttkon = dKttkon/(1.0f + gamma2);

      	dTkin = -2.0f*V*((Tkin + T0 + T0)/(Tkin + T0))*Tkin*dt/(3.0f*r); 

		// drift parameters
		Bfield = A*sqrtf((1.0f + gamma2))/(r*r); 
		Larmor = 0.0225f*Rig/Bfield;     
		alphaH = Pi / sinf(alphaM + (2.0f*Larmor*Pi/(r*180.0f)));   // PREVERIT v Burgerovom clanku 
		alphaH = alphaH * -1.0;
		alphaH = 1.0f/alphaH;
		alphaH = acosf(alphaH);
		arg = (1.0f-(2.0f*theta/Pi))*tanf(alphaH);
		f = atanf(arg)/alphaH;
		fprime = 1.0f+(arg*arg);
		fprime = tanf(alphaH)/fprime;
		fprime = -1.0f*fprime*2.0f/(Pi*alphaH);
		gamma2PlusOne2 = (1.0f + gamma2) * (1.0f + gamma2);

		// drift 
		DriftR = polarity*konvF*(2.0f/(3.0f*A))*Rig*beta*r*cosineTheta*gamma*f/((gamma2PlusOne2)*sineTheta);
		DriftSheetR = polarity*konvF*(1.0f/(3.0f*A))*Rig*beta*r*gamma*fprime/(1.0f + gamma2); 
		dr = dr + ((DriftR + DriftSheetR)*dt);
		DriftTheta = driftThetaConstant*Rig*beta*r*(2.0f+(gamma2))*f/(gamma2PlusOne2);
		dtheta += (DriftTheta*dt/r);

		// add calculated parameters to simulation variables
		previousTheta = theta;
		r = r + dr;
        theta = theta + dtheta;
		Tkin = Tkin - dTkin; 
        
		if (theta<0.0f) {
			theta = fabsf(theta);
		}
		if (theta>2.0f*Pi) {
			theta = theta - (2.0f*Pi);
		}
		else if (theta>Pi) {
			theta = (2.0f*Pi) - theta;
		}
		
		beta = sqrtf(Tkin*(Tkin + T0 + T0)) / (Tkin + T0);

		if (r<0.1f) {
        	r -= dr;
    		theta = previousTheta; 
			Tkin += dTkin;
		}
        
		if (beta > 0.001f && Tkin < 200.0f && r > 100.0f) {
			count = atomicAdd(&outputCounter, 1);
			history[count].setValues(Tkin, r, id, theta);
			break;
		}
		else if (beta<0.01f) {
			break;
		}


	}
	state[id] = cuState[idx];
}

void runTwoDimensionBpMethod(simulationInputTwoDimensionBP *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
	}
	if (!createDirectory("2DBP", destination))
	{
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->threadSize == 1024)
	{
		cudaFuncSetAttribute(trajectorySimulationTwoDimensionBp, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
	}
	for (int k = 0; k < iterations; ++k)
	{
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		wCalc<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, simulation->w, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulationTwoDimensionBp<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->pinj, simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryTwoDimensionBP), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, "%g %g %g %g %g %g\n", simulation->Tkininj[simulation->local_history[j].id], simulation->local_history[j].Tkin, simulation->local_history[j].r, simulation->w[simulation->local_history[j].id], 3.1415926535f / 2.0f, simulation->local_history[j].theta);
			}
		}
	}
	fclose(file);
}