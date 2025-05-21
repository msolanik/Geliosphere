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
	for (int iteration = 0; iteration < targetIterations; iteration++)
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

	auto& gen = generator; 
    auto& dist = distribution;

	std::vector<SimulationOutput> localOutputs;
	localOutputs.reserve(101);

	for (int energy = 0; energy < 101; energy++) {
		for (int particlePerEnergy = 0; particlePerEnergy < 250; particlePerEnergy++) {
			Tkininj = getTkinInjection(((availableThreads * iteration + threadNumber) * 250) + particlePerEnergy, 0.0001, uniformEnergyInjectionMaximum, 10000);
			Tkin = Tkininj;

			Tkinw = Tkin * 1e9 * q;						
			Rig = RigFromTkinJoule(Tkin);
			p = Rig * q / c;
			pinj = p;

			w = W(p);
			sumac = 0.0;
			r = 100.0001;

			while (r < 100.0002) {
				beta = Beta(Tkin);
				Rig = RigFromTkin(Tkin);
				Kdiff = Kdiffr(beta, Rig);
				arnum = dist(gen);
				//arnum = distribution(generator);
				dr = Dr(V, Kdiff, r, dt, arnum);
				dp = Dp(V, p, r);
				cfactor = Cfactor(V, r);
				sumac += cfactor * dt;

				rp = r;
				pp = p;

				r += dr;
				p -= dp;

				Rig = p * c / q;
				Tkin = TkinFromRig(Rig);

				if (beta > 0.01 && Tkin < 100.0) {
					if ((r - 1.0) / ((r - dr) - 1.0) < 0.0) {
						//outputMutex.lock();
						//outputQueue.push({pinj, p, r, w, sumac});
						localOutputs.emplace_back(SimulationOutput{pinj, p, r, w, sumac});
						//outputMutex.unlock();
					}
				}
				if (beta < 0.01) break;
				if (r < 0.1) { r = rp; p = pp; }
			}
		}
	}
	
	
	std::lock_guard<std::mutex> lock(outputMutex);
	for (const auto& output : localOutputs)
	{
		outputQueue.push(output);
	}
	 
}

double OneDimensionFpCpuModel::Beta(double Tkin)
{
	return sqrt(Tkin * (Tkin + 2.0 * T0)) / (Tkin + T0);
}

double OneDimensionFpCpuModel::RigFromTkin(double Tkin)
{
	return sqrt(Tkin * (Tkin + 2.0 * T0));
}

double OneDimensionFpCpuModel::RigFromTkinJoule(double Tkin)
{
	double Tkinw = Tkin * 1e9 * q;
	return sqrt(Tkinw * (Tkinw + 2.0 * T0w)) / q;
}

double OneDimensionFpCpuModel::Kdiffr(double beta, double Rig)
{
	return K0 * beta * Rig;
}

double OneDimensionFpCpuModel::Dp(double V, double p, double r)
{
	return 2.0 * V * p * dt / (3.0 * r);
}

double OneDimensionFpCpuModel::Dr(double V, double Kdiff, double r, double dt, double rand)
{
	return (V + 2.0 * Kdiff / r) * dt + (rand * sqrt(2.0 * Kdiff * dt));
}

double OneDimensionFpCpuModel::Cfactor(double V, double r)
{
	return 4.0 * V / (3.0 * r);
}

double OneDimensionFpCpuModel::TkinFromRig(double Rig)
{
	return (sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9)) / (q * 1e9);
}

double OneDimensionFpCpuModel::W(double p)
{
	double w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
	return pow(w, -1.85) / p / 1e45;
}