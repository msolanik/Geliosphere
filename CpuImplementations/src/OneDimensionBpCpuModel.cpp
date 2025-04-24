/*
unoptimized duration: 35 min with O3 flag for: time ./Geliosphere -B -d 5 -N 1

32m20s

30m17s

*/
#include "OneDimensionBpCpuModel.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

// Constants to avoid repeated calls
const double C_Q = c / q;
const double C_Q_1E9 = C_Q / 1e9;


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

    // cache TLS references to avoid __tls_get_addr overhead
    auto& gen = generator; 
    auto& dist = distribution; //avoids repeated __tls_get_addr lookup for thread-local variables



	for (int energy = 0; energy < 101; energy++)
	{
		for (int particlePerEnergy = 0; particlePerEnergy < 250; particlePerEnergy++)
		{
			Tkininj = getTkinInjection(((availableThreads * iteration + threadNumber) * 250) + particlePerEnergy, 0.0001, uniformEnergyInjectionMaximum, 10000);
			Tkin = Tkininj;

			Rig = RigFromTkin(Tkin);
			//p = Rig * 1e9 * q / c;
			p = Rig / C_Q_1E9;
			r = rInit;

			while (r < 100.0002)
			{
				beta = Beta(Tkin);
				//Rig = (p * c / q) / 1e9;
				Rig = p * C_Q_1E9;
				pp = p;
				dp = Dp(V, pp, r);
				p -= dp;

				Kdiff = Kdiffr(beta, Rig);
				arnum = dist(gen);  // much faster than distribution(generator);
				//arnum = distribution(generator);
				dr = Dr(V, Kdiff, r, dt, arnum);
				r += dr;

				// Rig = p * c / q;
				Rig = p * C_Q;
				//Tkin = TkinFromRig(Rig);
				Tkin = (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
				beta = Beta(Tkin);

				if (beta > 0.01f && Tkin < 200.0f)
				{
					if ((r > 100.0f) && ((r - dr) < 100.0f))
					{
						w = W(p);
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

double OneDimensionBpCpuModel::Beta(double Tkin)
{
    return sqrt(Tkin * (Tkin + 2.0 * T0)) / (Tkin + T0);
}

double OneDimensionBpCpuModel::RigFromTkin(double Tkin)
{
    return sqrt(Tkin * (Tkin + 2.0 * T0));
}

//double OneDimensionBpCpuModel::RigFromMomentum(double p)
//{
//   return (p * c / q) / 1e9;
//}

double OneDimensionBpCpuModel::Kdiffr(double beta, double Rig)
{
    return K0 * beta * Rig;
}

double OneDimensionBpCpuModel::Dp(double V, double p, double r)
{
    return (2.0 * V * p * dt / (3.0 * r));
}

double OneDimensionBpCpuModel::Dr(double V, double Kdiff, double r, double dt, double rand)
{
    return (V + (2.0 * Kdiff / r)) * dt + (rand * sqrt(2.0 * Kdiff * dt));
}

//double OneDimensionBpCpuModel::TkinFromRig(double Rig)
//{
//    return (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
//}

double OneDimensionBpCpuModel::W(double p)
{
    double w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
    return (pow(w, -1.85) / p) / 1e45;
}
