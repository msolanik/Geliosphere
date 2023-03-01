#include "OneDimensionFpCpuSimulation.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void OneDimensionFpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
{
	spdlog::info("Starting initialization of 1D F-p simulation.");
	srand(time(NULL));
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
		spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("FW", destination))
	{
		spdlog::error("Directory for 1D F-p simulations cannot be created.");
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
			threads.emplace_back(std::thread(&OneDimensionFpCpuSimulation::simulation, this, i, nthreads, mmm));
		}
		for (auto &th : threads)
		{
			th.join();
		}
		while (!outputQueue.empty())
		{
			struct SimulationOutput simulationOutput = outputQueue.front();
			fprintf(file, " %g  %g  %g  %g %g \n", simulationOutput.pinj, simulationOutput.p, simulationOutput.r, simulationOutput.w, simulationOutput.sumac);
			outputQueue.pop();
		}
	}
	fclose(file);
	writeSimulationReportFile(singleTone);
}

void OneDimensionFpCpuSimulation::simulation(int threadNumber, unsigned int availableThreads, int iteration)
{
	double r, K, dr, arnum;
	double Tkin, Tkininj, Rig, tt, t2, beta;
	double w;
	double Tkinw, p, rp, dp, pp, pinj, cfactor, sumac;
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

			Tkinw = Tkin * 1e9 * q;						
			Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q;
			p = Rig * q / c;
			pinj = p;

			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			sumac = 0.0;
			r = 100.0001;

			while (r < 100.0002)
			{
				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

				K = K0 * beta * Rig;

				arnum = distribution(generator);

				dr = (V + (2.0 * K / r)) * dt;
				dr = dr + (arnum * sqrt(2.0 * K * dt));

				dp = 2.0 * V * p * dt / (3.0 * r);

				cfactor = 4.0 * V / (3.0 * r);
				sumac = sumac + (cfactor * dt);

				rp = r;
				pp = p;

				r = r + dr;
				p = p - dp;

				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				Rig = p * c / q;
				Tkin = sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9);
				Tkin = Tkin / (q * 1e9); 

				if (beta > 0.01)
				{
					if (Tkin < 100.0)
					{
						if ((r - 1.0) / ((r - dr) - 1.0) < 0.0)
						{
							outputMutex.lock();
							outputQueue.push({pinj, p, r, w, sumac});
							outputMutex.unlock();
						}
					}
				}

				if (beta < 0.01)
				{
					break;
				}
				if (r < 0.1)
				{
					r = rp;
					p = pp;
				}
			} 
		}	  
	}		  
}