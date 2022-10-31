#include "TwoDimensionBpCpuSimulation.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void TwoDimensionBpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
{
	spdlog::info("Starting initialization of 2D B-p simulation.");
	srand(time(NULL));
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
	unsigned int nthreads = std::thread::hardware_concurrency();
	int new_MMM = ceil(singleTone->getInt("millions", 1) * 1000000 / (nthreads * 30 * 10000));
	setContants(singleTone);
	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) mmm / ((float) new_MMM / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&TwoDimensionBpCpuSimulation::simulation, this));
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
}

void TwoDimensionBpCpuSimulation::simulation()
{
	double r, K, dr, arnum, theta, thetainj, Kpar, Bfactor, dtem1;
	double Tkin, Tkininj, Rig, tt, t2, beta, alfa, Ktt, dKrr;
	double w, r2, gammma, gamma2, tmp1, tem2, Kper, Krr, dKtt;
	double Tkinw, p, dp, pinj, cfactor, sumac;
	int m, mm;
	double dtheta, dKttkon, dTkin;
	double DriftR,DriftTheta,arg,alphaH,Larmor,Bfield,f,fprime,DriftSheetR;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (m = 0; m < 30; m++)
	{
		for (mm = 0; mm < 10000; mm++)
		{

			Tkininj = SPbins[m];
			Tkin = Tkininj;

			Tkinw = Tkin * 1e9 * q;
			Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q;
			p = Rig * q / c;
			pinj = p;

			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			r = 1.0;
			theta = Pi/2.;
			thetainj = theta;

			while (r < 100.0)
			{
				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				alfa = t2/tt;
				
				Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

				r2 = r * r;

				gammma = (r * omega) * sin(theta) / V;
       			gamma2 = gammma*gammma;
       			tmp1 = 1.0 + gamma2;
       			tem2 = tmp1 * tmp1;

				Bfactor = (5.0/3.4) *  r2 / sqrt(tmp1);

				Kpar = K0*beta*Rig*Bfactor/3.0;
				if (Rig<0.1)
				{
					Kpar = K0*beta*0.1*Bfactor/3.0;
				}
				Kper = ratio * Kpar;

				Krr = Kper + ((Kpar - Kper)/tmp1);
				Ktt = Kper;

				dtem1 = 2.0*r*omega*omega*sin(theta)*sin(theta)/(V*V);
				dKrr = ratio*K0*beta*Rig*((2.0*r*sqrt(tmp1)) - (r2*dtem1/(2.0*sqrt(tmp1))))/tmp1;
				dKrr = dKrr +  ((1.0-ratio)*K0*beta*Rig*((2.0*r*pow(tmp1,1.5))-(r2*dtem1*3.0*sqrt(tmp1)/2.0))/pow(tmp1,3.0));
				dKrr = dKrr*5.0/(3.0*3.4);
				
				dKtt = (-1.0*ratio*K0*beta*Rig*r2*sin(theta)*cos(theta)*(omega*omega*r2/(V*V)))/pow(tmp1,1.5);

				dr = ((-1.0*V) + (2.0*Krr/r) + dKrr)*dt;
				dr = dr + (distribution(generator)*sqrt(2.0*Krr*dt));

				dtheta = (Ktt*cos(theta)) / (r2 * sin(theta));
				dtheta = dtheta*dt/tmp1;
				dtheta = dtheta +  ((distribution(generator)*sqrt(2.0*Ktt*dt)) / r);

				dKttkon = (Ktt*cos(theta)) / (r2 * sin(theta));
				dKttkon = dKttkon +  (dKtt/r2); 

				dKttkon = (Ktt*cos(theta)) / (r2 * sin(theta));
				dKttkon = dKttkon/tmp1;

				dTkin = -2.0*V*alfa*Tkin*dt/(3.0*r);  
				
				Bfield = A*sqrt(tmp1)/(r*r);

				Larmor = 0.0225*Rig/Bfield;

				alphaH = Pi / sin(alphaM + (2.0*Larmor*Pi/(r*180.0)));
				alphaH = alphaH -1.0;
				alphaH = 1.0/alphaH;
				alphaH = acos(alphaH);;

				arg = (1.-(2.*theta/Pi))*tan(alphaH);
				f = atan(arg)/alphaH;

				DriftR = polarity*konvF*(2.0/(3.0*A))*Rig*beta*r*cos(theta)*gammma*f/(tem2*sin(theta));				

				fprime = 1.0+(arg*arg);
				fprime = tan(alphaH)/fprime;
				fprime = -1.0*fprime*2.0/(Pi*alphaH);
				DriftSheetR = polarity*konvF*(1.0/(3.0*A))*Rig*beta*r*gammma*fprime/tmp1; 
				DriftTheta = -1.0*polarity*konvF*(2.0/(3.0*A))*Rig*beta*r*gammma*(2.0+(gammma*gammma))*f/tem2;

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