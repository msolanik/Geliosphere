#include "OneDimensionBpCpuModel.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void OneDimensionBpCpuModel::runSimulation(ParamsCarrier *singleTone)
{
	srand(time(NULL));
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
	unsigned int nthreads = std::thread::hardware_concurrency();
	int targetIterations = ceil((double)singleTone->getInt("millions", 1) * 1000000.0 / ((double)nthreads * 101.0 * 250.0));
	setContants(singleTone);
	for (int iteration = 0; iteration < targetIterations; iteration++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) iteration / ((float) targetIterations / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&OneDimensionBpCpuModel::simulation, this, i, nthreads, iteration));
		}
		for (auto &th : threads)
		{
			th.join();
		}
		spdlog::info("In this iteration {} particles were detected.", outputQueue.size());
		while (!outputQueue.empty())
		{
			struct SimulationOutput simulationOutput = outputQueue.front();
            fprintf(file, "%g %g %g %g\n", simulationOutput.Tkin, simulationOutput.Tkininj, simulationOutput.r, simulationOutput.w);
			outputQueue.pop();
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
}

void OneDimensionBpCpuModel::simulation(int threadNumber, unsigned int availableThreads, int iteration)
{
	double r, dr, arnum;
	double Tkin, Tkininj, Rig, beta;
	double w;
	double p, dp, pp, Kdiff;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (int energy = 0; energy < 101; energy++)
	{
		for (int particlePerEnergy = 0; particlePerEnergy < 250; particlePerEnergy++)
		{
			Tkininj = getTkinInjection(((availableThreads * iteration + threadNumber) * 250) + particlePerEnergy, 0.0001, uniformEnergyInjectionMaximum, 10000);
			Tkin = Tkininj;

			Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));
			p = Rig * 1e9 * q / c;		
			r = rInit;

			while (r < 100.0002)
			{
				// Equation 5
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				// Equation 6 in GeV
				Rig = (p * c / q) / 1e9;
                pp = p;
                
				// Equation 14
				dp = (2.0f * V * pp * dt / (3.0f * r));
				p -= dp;

				// Equation 7
				Kdiff = K0 * beta * Rig;
                arnum = distribution(generator);
                
				// Equation 13
				dr = (V + (2.0 * Kdiff / r)) * dt + (arnum * sqrt(2.0 * Kdiff * dt));
			    r += dr;

				// Equation 6 in J
                Rig = p * c / q;
                Tkin = (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
                
				// Equation 5
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				if (beta > 0.01f && Tkin < 200.0f)
		        {
			        if ((r > 100.0f) && ((r - dr) < 100.0f))
			        {
						// Equation under Equation 6 from 
            			// Yamada et al. "A stochastic view of the solar modulation phenomena of cosmic rays" GEOPHYSICAL RESEARCH LETTERS, VOL. 25, NO.13, PAGES 2353-2356, JULY 1, 1998
            			// https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL51869
						w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
						w = (pow(w, -1.85) / p) / 1e45;
						
						outputMutex.lock();
						outputQueue.push({Tkin, Tkininj, r, w});
						outputMutex.unlock();
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
		}	  
	}		  
}