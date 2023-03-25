#include "OneDimensionFpCpuModel.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void OneDimensionFpCpuModel::runSimulation(ParamsCarrier *singleTone)
{
	spdlog::info("Starting initialization of 1D F-p simulation.");
	srand(time(NULL));
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
		spdlog::info("Destination is not specified - using generated name for destination: " + destination);
	}
	if (!createDirectory("1DFP", destination))
	{
		spdlog::error("Directory for 1D F-p simulations cannot be created.");
		return;
	}

	FILE *file = fopen("log.dat", "w");
	unsigned int nthreads = std::thread::hardware_concurrency();
	int targetIterations = ceil((double)singleTone->getInt("millions", 1) * 1000000.0 / ((double)nthreads * 101.0 * 250.0));
	setContants(singleTone);
	for (int iteration = 0; targetIterations < targetIterations; iteration++)
	{
		spdlog::info("Processed: {:03.2f}%", (float) iteration / ((float) targetIterations / 100.0));
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&OneDimensionFpCpuModel::simulation, this, i, nthreads, iteration));
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

void OneDimensionFpCpuModel::simulation(int threadNumber, unsigned int availableThreads, int iteration)
{
	double r, Kdiff, dr, arnum;
	double Tkin, Tkininj, Rig, beta;
	double w;
	double Tkinw, p, rp, dp, pp, pinj, cfactor, sumac;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (int energy = 0; energy < 101; energy++)
	{
		for (int particlePerEnergy = 0; particlePerEnergy < 250; particlePerEnergy++)
		{
			Tkininj = getTkinInjection(((availableThreads * iteration + threadNumber) * 250) + particlePerEnergy, 0.0001, uniformEnergyInjectionMaximum, 10000);
			Tkin = Tkininj;

			Tkinw = Tkin * 1e9 * q;						
			Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q;
			p = Rig * q / c;
			pinj = p;

			// Equation under Equation 6 from 
            // Yamada et al. "A stochastic view of the solar modulation phenomena of cosmic rays" GEOPHYSICAL RESEARCH LETTERS, VOL. 25, NO.13, PAGES 2353-2356, JULY 1, 1998
            // https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL51869
			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			sumac = 0.0;
			r = 100.0001;

			while (r < 100.0002)
			{
				// Equation 5
				beta = sqrt(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

				// Equation 6
				Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

				// Equation 7
				Kdiff = K0 * beta * Rig;

				arnum = distribution(generator);

				// Equation 9
				dr = (V + (2.0 * Kdiff / r)) * dt;
				dr = dr + (arnum * sqrt(2.0 * Kdiff * dt));

				// Equation 10
				dp = 2.0 * V * p * dt / (3.0 * r);

				// Equation 11
				cfactor = 4.0 * V / (3.0 * r);
				sumac = sumac + (cfactor * dt);

				rp = r;
				pp = p;

				r = r + dr;
				p = p - dp;

				// Equation 6 in J
				Rig = p * c / q;
				Tkin = sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9);
				Tkin = Tkin / (q * 1e9); 

				if (beta > 0.01 & Tkin < 100.0)
				{
					if ((r - 1.0) / ((r - dr) - 1.0) < 0.0)
					{
						outputMutex.lock();
						outputQueue.push({pinj, p, r, w, sumac});
						outputMutex.unlock();
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