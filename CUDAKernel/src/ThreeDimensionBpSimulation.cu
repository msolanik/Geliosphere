/**
 * @file ThreeDimensionBpSimulation.cu
 * @author Michal Solanik
 * @brief Implementation of 3D B-p method.
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
#include "ThreeDimensionBpSimulation.cuh"
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
__global__ void wCalcThreeBp(float *Tkininj, float *p, double *w, int padding) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    Tkininj[id] = getSolarPropInjection(BLOCK_SIZE_THREE_BP * THREAD_SIZE_THREE_BP * padding + id);
	float Tkinw = Tkininj[id]*1e9f*q;
	float Rig = sqrtf(Tkinw*(Tkinw + (2.0f * T0w))) / q;
	float pinj = Rig*q / c;
	double newW = (m0_double*m0_double*c_double*c_double*c_double*c_double) + (pinj*pinj*c_double*c_double);
	newW = (pow(newW, -1.85) / pinj) / 1e45;
	w[id] = newW;
	p[id] = pinj;
}

/**
 * @brief Run simulations for 3D B-p method.
 *
 * @param history Data structure containing output records.
 * @param padding Support value used to calculate state for getting
 * kinetic energy.
 * @param state Array of random number generator data structures.
 */
__global__ void trajectorySimulationThreeDimensionBp(trajectoryHistoryThreeDimensionBP* history, int padding, curandState* state) 
{
    extern __shared__ int sharedMemory[];
	int id = blockIdx.x * blockDim.x + threadIdx.x; 
	int idx = threadIdx.x;
	float r =  1.0f;
    float beta, Rig, dtem1, dKtt, dr, Krr, Bfactor, Kpar, dKrr, Ktt, Kper, dTkin, gamma, gamma2, dKttkon, tmp1, tem2;
	float Tkin = getSolarPropInjection(BLOCK_SIZE_THREE_BP * THREAD_SIZE_THREE_BP * padding + id);
	float dtheta, theta;
	float Bfield, Larmor, alphaH, arg, f, fprime, DriftR, DriftTheta, DriftSheetR;
	float delta, deltarh, deltarh2;
	float Kphph, Krt, Krph, Ktph;
	float B111, B11, B12, B13, B22, B23;
	float sin3, sin2, dKtt0, dKtt1, dKtt2, dKrtr, dKrtt;
	theta = thetainj;
    float2 *generated = (float2 *)sharedMemory;
	curandState *cuState = (curandState *)(&generated[THREAD_SIZE_THREE_BP]);
	cuState[idx] = state[blockIdx.x * blockDim.x + threadIdx.x];
	int count;
	while (r < 100.0f) 
    {
		beta = sqrtf(Tkin*(Tkin + T0 + T0)) / (Tkin + T0);
		Rig = sqrtf(Tkin*(Tkin + (2.0f * T0)));

        if ((theta < (1.7f * Pi / 180.0f)) || (theta > (178.3f * Pi / 180.0f)))
        {
        	delta = 0.2f * 0.003f;
        }
		else 
		{
            delta = delta0 / sinf(theta);
		}
		deltarh = delta / rh;
		deltarh2 = deltarh * deltarh;
        		
		gamma = (r * omega) * sinf(theta) / V; 
        gamma2 = gamma * gamma;     
		tmp1 = 1.0f + gamma2 + (r * r * deltarh2);
        tem2 = tmp1 * tmp1;
        Bfactor = (5.0f / 3.4f) * r * r / sqrtf(tmp1);

        if (Rig < 0.1f)
        {
        	Kpar = K0 * beta * 0.1f * Bfactor / 3.0f;
        }
        else 
        {
            Kpar = K0 * beta * Rig * Bfactor / 3.0f;
        }        

		Kper = ratio * Kpar;

        Krr = Kper + ((Kpar - Kper) / tmp1);
        Ktt = Kper + (r * r * deltarh2 * (Kpar - Kper) / tmp1);
		Kphph = 1.0f;

        Krt = deltarh * (Kpar - Kper) * r / tmp1;
        Krph = 0.0f;
		Ktph = 0.0f;
		
		B111 = (Kphph * Krt * Krt) - (2.0f * Krph * Krt * Ktph) + (Krr * Ktph * Ktph) + (Ktt * Krph * Krph) - (Krr * Ktt * Kphph);
        B11 = 2.0f * B111 / ((Ktph * Ktph) - (Ktt * Kphph));
        B11 = sqrtf(B11);
        B12 = ((Krph * Ktph) - (Krt * Kphph)) / ((Ktph * Ktph) - (Ktt * Kphph));
        B12 = B12 * sqrtf(2.0f * (Ktt - (Ktph * Ktph / Kphph)));
        B13 = sqrtf(2.0f) * Krph / sqrtf(Kphph);
        B22 = Ktt - (Ktph * Ktph / Kphph);
        B22 = sqrtf(2.0f * B22) / r;
        B23 = Ktph * sqrtf(2.0f / Kphph) / r;

		dtem1 = 2.0f * r * omega * omega * sinf(theta) * sinf(theta) / (V * V);
        dtem1 = dtem1 + (2.0f * r * deltarh2);                                                                 
        dKrr = ratio * K0 * beta * Rig * ((2.0f * r * sqrtf(tmp1)) - (r * r * dtem1 / (2.0f * sqrtf(tmp1)))) / tmp1;
        dKrr = dKrr + ((1.0f - ratio) * K0 * beta * Rig * ((2.0f * r * powf(tmp1, 1.5f)) - (r * r * dtem1 * 3.0f * sqrtf(tmp1) / 2.0f)) / powf(tmp1, 3.0f));
        dKrr = dKrr * 5.0f / (3.0f * 3.4f);
		
		dr = ((-1.0f*V) + (2.0f*Krr/r) + dKrr)*dt;
       	generated[idx] = curand_box_muller(&cuState[idx]);
		dr = dr + (generated[idx].x * B11 * sqrtf(dt));
    	dr = dr + (generated[idx].y * B12 * sqrtf(dt));
        dr = dr + (curand_normal(&cuState[idx]) * B13 * sqrtf(dt));

		sin3 = sinf(theta)*sinf(theta)*sinf(theta);

        dKtt = sinf(theta)*cosf(theta)*(omega*omega*r*r/(V*V));
        dKtt = dKtt - (r*r*delta0*delta0*cosf(theta)/(rh*rh*sin3));
        dKtt = (-1.0f*ratio*K0*beta*Rig*r*r*dKtt)/powf(tmp1,1.5f);

        if ((theta>(1.7f*Pi/180.0f))&&(theta<(178.3f*Pi/180.0f))) 
        {
        	dKtt0 = 3.0f*(1.0f-ratio)*K0*beta*Rig*r*r*r*r*deltarh2;
            dKtt1 = omega*omega*r*r*sinf(theta)*cosf(theta)/(V*V*powf(tmp1,2.5f));
            dKtt = dKtt - (dKtt0*dKtt1);
        }
        else
        {
            sin2 = sinf(theta)*sinf(theta);
            dKtt0 = (1.0f-ratio)*K0*beta*Rig*r*r*r*r*delta0*delta0/(rh*rh);
            dKtt1 = -2.0f*(cosf(theta)/sin3)/powf(tmp1,1.5f);
            dKtt2 = 1.5f*((2.0f*omega*omega*r*r*sinf(theta)*cosf(theta)/(V*V)) - (2.0f*r*r*delta0*delta0*cosf(theta)/(rh*rh*sin3)))/(sin2*powf(tmp1,2.5f));
            dKtt = dKtt + (dKtt0*(dKtt1 - dKtt2));
        }

		dKrtr = (1.0f - ratio) * K0 * beta * Rig * deltarh * 3.0 * r * r / powf(tmp1, 2.5f);

        if ((theta > (1.7f * Pi / 180.0f)) && (theta < (178.3f * Pi / 180.0f)))
        {
        	dKrtt = (1.0f - ratio) * K0 * beta * Rig * r * r * r / (rh * powf(tmp1, 2.5f));
            dKrtt = -1.0f * dKrtt * delta;
            dKrtt = dKrtt * 3.0f * gamma2 * cosf(theta) / sinf(theta);
        }
        else
        {
            dKrtt = (1.0f - ratio) * K0 * beta * Rig * r * r * r / (rh * powf(tmp1, 2.5f));
            dKrtt = -1.0f * dKrtt * delta0 * cosf(theta) / (sinf(theta) * sinf(theta));
            dKrtt = dKrtt*(1.0f - (2.0f*r*r*deltarh2) + (4.0f*gamma2)); 
        }

       	dr = dr + (dKrtt*dt/r) + (Krt*cosf(theta)*dt/(r*sinf(theta)));

        dtheta = (Ktt * cosf(theta)) / (r * r * sinf(theta));
        dtheta = (dtheta*dt) + (dKtt*dt/(r*r));
        dtheta = dtheta + (dKrtr * dt) + (2.0f * Krt * dt / r);                                                    
		generated[idx] = curand_box_muller(&cuState[idx]);
        dtheta = dtheta + (generated[idx].x * B22 * sqrtf(dt)) + (generated[idx].y * B23 * sqrtf(dt));

		dKttkon = (Ktt * cosf(theta)) / (r * r * sinf(theta));
        dKttkon = dKttkon / tmp1;

        dTkin = -2.0f * V * ((Tkin + T0 + T0)/(Tkin + T0)) * Tkin * dt / (3.0f * r);

        Bfield = A * sqrtf(tmp1) / (r * r);
    	Larmor = 0.0225f * Rig / Bfield;

        alphaH = Pi / sinf(alphaM + (2.0f * Larmor * Pi / (r * 180.0f)));
        alphaH = alphaH - 1.0f;
        alphaH = 1.0f / alphaH;
        alphaH = acosf(alphaH);

        arg = (1.0f - (2.0f * theta / Pi)) * tanf(alphaH);
        f = atanf(arg) / alphaH;
		DriftR = polarity * konvF * (2.0f / (3.0f * A)) * Rig * beta * r * cosf(theta) * gamma * f / (tem2 * sinf(theta));
        DriftTheta = -1.0f * polarity * konvF * (2.0f / (3.0f * A)) * Rig * beta * r * gamma * (2.0f + (gamma * gamma)) * f / tem2;
        fprime = 1.0f + (arg * arg);
        fprime = tanf(alphaH) / fprime;
        fprime = -1.0f * fprime * 2.0f / (Pi * alphaH);

        DriftSheetR = polarity * konvF * (1.0f / (3.0f * A)) * Rig * beta * r * gamma * fprime / tmp1;
        dr = dr + ((DriftR + DriftSheetR) * dt);
        dtheta = dtheta + (DriftTheta * dt / r);
		theta = theta + dtheta;
        if (theta < 0.0f)
        {
        	theta = fabs(theta);
        }
        if (theta > 2.0f * Pi)
        {
        	theta = theta - (2.0f * Pi);
        }
        else if (theta > Pi)
        {
        	theta = (2.0f * Pi) - theta;
        }
        r = r + dr;
        Tkin = Tkin - dTkin; 
        
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
 * @brief Run 3D B-p method with given parameters defines 
 * in input simulation data structure.
 * 
 */
void runThreeDimensionBpMethod(simulationInputThreeDimensionBP *simulation)
{
	int counter;
	ParamsCarrier *singleTone;
	singleTone = simulation->singleTone;
    spdlog::info("Starting initialization of 3D B-p simulation.");

	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
        spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("3DBP", destination))
	{
        spdlog::error("Directory for 3D B-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	curandInitialization<<<simulation->blockSize, simulation->threadSize>>>(simulation->state);
	gpuErrchk(cudaDeviceSynchronize());
	int iterations = ceil((float)singleTone->getInt("millions", 1) * 1000000  / ((float)simulation->blockSize * (float)simulation->threadSize));
	if (simulation->threadSize == 1024)
	{
		cudaFuncSetAttribute(trajectorySimulationThreeDimensionBp, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
	}
	for (int k = 0; k < iterations; ++k)
	{
        spdlog::info("Processed: {:03.2f}%", (float)k / ((float)iterations / 100.0));
		nullCount<<<1, 1>>>();
		gpuErrchk(cudaDeviceSynchronize());
		wCalcThreeBp<<<simulation->blockSize, simulation->threadSize>>>(simulation->Tkininj, simulation->pinj, simulation->w, k);
		gpuErrchk(cudaDeviceSynchronize());
		trajectorySimulationThreeDimensionBp<<<simulation->blockSize, simulation->threadSize, simulation->threadSize * sizeof(curandState_t) + simulation->threadSize * sizeof(float2)>>>(simulation->history, k, simulation->state);
		gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpyFromSymbol(&counter, outputCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        spdlog::info("In this iteration {} particles were detected.", counter);
		if (counter != 0)
		{
			gpuErrchk(cudaMemcpy(simulation->local_history, simulation->history, counter * sizeof(trajectoryHistoryThreeDimensionBP), cudaMemcpyDeviceToHost));
			for (int j = 0; j < counter; ++j)
			{
				fprintf(file, "%g %g %g %g %g %g\n", simulation->Tkininj[simulation->local_history[j].id], simulation->local_history[j].Tkin, simulation->local_history[j].r, simulation->w[simulation->local_history[j].id], 3.1415926535f / 2.0f, simulation->local_history[j].theta);
			}
		}
	}
	fclose(file);	
    spdlog::info("Simulation ended.");
}