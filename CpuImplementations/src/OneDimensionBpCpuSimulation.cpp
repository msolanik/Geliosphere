#include "OneDimensionBpCpuSimulation.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void OneDimensionBpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
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
	int new_MMM = ceil(singleTone->getInt("millions", 1) * 1000000 / (nthreads * 101 * 10000));
	setContants(singleTone, true);
	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) mmm / ((float) new_MMM / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&OneDimensionBpCpuSimulation::simulation, this));
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
}

void OneDimensionBpCpuSimulation::simulation()
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
		for (mm = 0; mm < 10000; mm++)
		{

			Tkininj = (m) + ((mm + 1) / 10000.0);
			Tkin = Tkininj;

			Rig = sqrt(Tkin * (Tkin + (2.0 * T0))); /*vo Voltoch*/
			p = Rig * 1e9 * q / c;		
			r = 1.0;

			while (r < 100.0002)
			{ /* one history */
				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				Rig = (p * c / q) / 1e9;
                pp = p;
                p -= (2.0f * V * pp * dt / (3.0f * r));

                arnum = distribution(generator);
                dr = (V + (2.0 * K0 * beta * Rig / r)) * dt + (arnum * sqrt(2.0 * K0 * beta * Rig * dt));
			    r += dr;

                Rig = p * c / q;
                Tkin = (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
                Rig = Rig / 1e9;
                beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				if (beta > 0.01f && Tkin < 200.0f)
		        {
			        if ((r > 100.0f) && ((r - dr) < 100.0f))
			        {
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