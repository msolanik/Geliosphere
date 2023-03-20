#include "SolarPropLikeCpuModel.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>
#include <cmath>

void SolarPropLikeCpuModel::runSimulation(ParamsCarrier *singleTone)
{
	spdlog::info("Starting initialization of SOLARPROPlike 2D simulation.");
	srand(time(NULL));
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
		spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("2DBP", destination))
	{
		spdlog::error("Directory for SOLARPROPlike 2D simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	unsigned int nthreads = std::thread::hardware_concurrency();
	int new_MMM = ceil((double)singleTone->getInt("millions", 1) * 1000000.0 / ((double)nthreads * 30.0 * 500.0));
	setContants(singleTone);
	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) mmm / ((float) new_MMM / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&SolarPropLikeCpuModel::simulation, this, i, nthreads, mmm));
		}
		for (auto &th : threads)
		{
			th.join();
		}
		spdlog::info("In this iteration {} particles were detected.", outputQueue.size());
		while (!outputQueue.empty())
		{
			struct SimulationOutput simulationOutput = outputQueue.front();
			fprintf(file, "%g %g %g %g %g %g\n", simulationOutput.Tkininj, simulationOutput.Tkin, simulationOutput.r, simulationOutput.w, simulationOutput.thetainj, simulationOutput.theta);
			outputQueue.pop();
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
}

void SolarPropLikeCpuModel::simulation(int threadNumber, unsigned int availableThreads, int iteration)
{
	double r, dr, arnum, theta, Kpar, Bfactor, dKrrTmp;
	double Tkin, Tkininj, Rig, beta, alfa, Ktt, dKrr;
	double w, r2, gamma, gamma2, onePlusGamma, onePlusGamma2, Kper, Krr;
	double Tkinw, p;
	int m, mm;
	double dtheta, dTkin;
	double DriftR,DriftTheta,arg,alphaH,Larmor,Bfield,f,fprime,DriftSheetR;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (m = 0; m < 30; m++)
	{
		for (mm = 0; mm < 500; mm++)
		{
			Tkininj = (useUniformInjection) 
				? getTkinInjection(((availableThreads * iteration + threadNumber) * 500) + mm, 0.0001, uniformEnergyInjectionMaximum, 10000)
				: SPbins[m];
			Tkin = Tkininj;

			Tkinw = Tkin * 1e9 * q;
			Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q;
			p = Rig * q / c;

			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			r = rInit;
			theta = thetainj;

			while (r < 100.0)
			{
				// Equation 42
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				alfa = (Tkin + T0 + T0)/(Tkin + T0);
				
				Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

				r2 = r * r;

                // Equation 13 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				gamma = (r * omega) * sin(theta) / V;
       			
				// Support variables
				gamma2 = gamma*gamma;
       			onePlusGamma = 1.0 + gamma2;
       			onePlusGamma2 = onePlusGamma * onePlusGamma;

				Bfactor = (5.0/3.4) *  r2 / sqrt(onePlusGamma);

                // Equation 42 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				if (Rig<0.1)
				{
					Kpar = K0*beta*0.1*Bfactor/3.0;
				}
				else 
				{
					Kpar = K0*beta*Rig*Bfactor/3.0;
				}

				// Equation 43 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				Kper = ratio * Kpar;

				// Equation 14 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				Krr = Kper + ((Kpar - Kper)/onePlusGamma);

				// Equation 15 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				Ktt = Kper;

				// Equation 44 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				dKrrTmp = 2.0*r*omega*omega*sin(theta)*sin(theta)/(V*V);
				dKrr = ratio*K0*beta*Rig*((2.0*r*sqrt(onePlusGamma)) - (r2*dKrrTmp/(2.0*sqrt(onePlusGamma))))/onePlusGamma;
				dKrr = dKrr +  ((1.0-ratio)*K0*beta*Rig*((2.0*r*pow(onePlusGamma,1.5))-(r2*dKrrTmp*3.0*sqrt(onePlusGamma)/2.0))/pow(onePlusGamma,3.0));
				dKrr = dKrr*5.0/(3.0*3.4);
				
				// Equation 16 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				dr = ((-1.0*V) + (2.0*Krr/r) + dKrr)*dt;
				dr = dr + (distribution(generator)*sqrt(2.0*Krr*dt));

				// Equation 17 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				dtheta = (Ktt*cos(theta)) / (r2 * sin(theta));
				dtheta = dtheta*dt/onePlusGamma;
				dtheta = dtheta +  ((distribution(generator)*sqrt(2.0*Ktt*dt)) / r);
				
				// Equation 18 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				dTkin = -2.0*V*alfa*Tkin*dt/(3.0*r);  
				
				Bfield = A*sqrt(onePlusGamma)/(r*r);
				
				// Equation 26 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				Larmor = 0.0225*Rig/Bfield;

                // Equation 34 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				alphaH = Pi / sin(alphaM + (2.0*Larmor*Pi/(r*180.0)));
				alphaH = alphaH -1.0;
				alphaH = 1.0/alphaH;
				alphaH = acos(alphaH);;

                // Equation 32 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				arg = (1.-(2.*theta/Pi))*tan(alphaH);
				f = atan(arg)/alphaH;

                // Equation 35 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				DriftR = polarity*konvF*(2.0/(3.0*A))*Rig*beta*r*cos(theta)*gamma*f/(onePlusGamma2*sin(theta));

				// Equation 36 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				DriftTheta = -1.0*polarity*konvF*(2.0/(3.0*A))*Rig*beta*r*gamma*(2.0+(gamma*gamma))*f/onePlusGamma2;
                
				// Equation 33 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				fprime = 1.0+(arg*arg);
				fprime = tan(alphaH)/fprime;
				fprime = -1.0*fprime*2.0/(Pi*alphaH);

				// Equation 37 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
				DriftSheetR = polarity*konvF*(1.0/(3.0*A))*Rig*beta*r*gamma*fprime/onePlusGamma; 

				dr = dr + ((DriftR + DriftSheetR)*dt);
				dtheta = dtheta + (DriftTheta*dt/r);

				r = r + dr;
				theta = theta + dtheta;

				if (theta<0.0) 
				{
					theta = fabs(theta);
				}
				if (theta>2.0*Pi) 
				{
					theta = theta - (2.0*Pi);
				}
				else if (theta>Pi)
				{
					theta = (2.0*Pi) - theta;
				}

				Tkin = Tkin - dTkin; 

				if (r < 0.0)
				{
					r = -1.0 * r;
				}
				if (r > 100.0)
				{
					outputMutex.lock();
					outputQueue.push({Tkininj,Tkin,r,w,thetainj,theta});
					outputMutex.unlock();
					break;
				}

			} 
		}	  
	}		  
}