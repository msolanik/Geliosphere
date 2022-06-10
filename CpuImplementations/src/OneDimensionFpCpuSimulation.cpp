#include "OneDimensionFpCpuSimulation.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include <thread>
#include <random>

void OneDimensionFpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
{
	srand(time(NULL));
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
	}
	if (!createDirectory("FW", destination))
	{
		return;
	}

	FILE *file = fopen("log.dat", "w");
	unsigned int nthreads = std::thread::hardware_concurrency();
	int new_MMM = ceil(1000 / nthreads) * singleTone->getInt("millions", 1);
	setContants(singleTone, false);
	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&OneDimensionFpCpuSimulation::simulation, this));
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
}

void OneDimensionFpCpuSimulation::simulation()
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

			Tkinw = Tkin * 1e9 * q;						 /*v Jouloch*/
			Rig = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q; /*vo Voltoch*/
			p = Rig * q / c;							 /*kg m s-1*/
			pinj = p;

			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			tt = Tkin + T0;
			t2 = tt + T0;
			beta = sqrt(Tkin * t2) / tt;

			sumac = 0.0;
			r = 100.0001;

			while (r < 100.0002)
			{ /* one history */
				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				Rig = sqrt(Tkin * (Tkin + (2 * T0)));

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

				Rig = p * c / q; /*Volts*/
				Tkin = sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9);
				Tkin = Tkin / (q * 1e9); /*GeV*/

				if (beta > 0.01)
				{
					if (Tkin < 100)
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
					r = 500;
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