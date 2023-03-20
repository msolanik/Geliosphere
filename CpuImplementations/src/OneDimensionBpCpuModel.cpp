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
	if (!createDirectory("BP", destination))
	{
		spdlog::error("Directory for 1D B-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	unsigned int nthreads = std::thread::hardware_concurrency();
	int new_MMM = ceil((double)singleTone->getInt("millions", 1) * 1000000.0 / ((double)nthreads * 101.0 * 250.0));
	setContants(singleTone);
	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) mmm / ((float) new_MMM / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&OneDimensionBpCpuModel::simulation, this, i, nthreads, mmm));
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
	int m, mm;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (m = 0; m < 101; m++)
	{
		for (mm = 0; mm < 250; mm++)
		{
			Tkininj = getTkinInjection(((availableThreads * iteration + threadNumber) * 250) + mm, 0.0001, uniformEnergyInjectionMaximum, 10000);
			Tkin = Tkininj;

			Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));
			p = Rig * 1e9 * q / c;		
			r = rInit;

			while (r < 100.0002)
			{
				// Equation 42
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				// Equation 43 in GeV
				Rig = (p * c / q) / 1e9;
                pp = p;
                
				// Equation 47
				dp = (2.0f * V * pp * dt / (3.0f * r));
				p -= dp;

				// Equation 44
				Kdiff = K0 * beta * Rig;
                arnum = distribution(generator);
                
				// Equation 47
				dr = (V + (2.0 * Kdiff / r)) * dt + (arnum * sqrt(2.0 * Kdiff * dt));
			    r += dr;

				// Equation 43 in J
                Rig = p * c / q;
                Tkin = (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
                
				// Equation 42
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				if (beta > 0.01f && Tkin < 200.0f)
		        {
			        if ((r > 100.0f) && ((r - dr) < 100.0f))
			        {
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