#include "TwoDimensionBpCpuSimulation.hpp"
#include "FileUtils.hpp"

#include <thread>
#include <random>

const double V = (8./3.)*1e-6;
const double dt = 1000.0;
const double K0 = 1.43831e-5;
const double N = 1e9;
const double m0 = 1.67261e-27;
const double q = 1.60219e-19;
const double c = 2.99793e8;
const double Pi = 3.1415926535897932384626433832795;
const double omega = 2.866e-6;
const double T0 = m0 * c * c / (q * 1e9);
const double T0w = m0 * c * c;
const double ratio = 0.02;
const double alphaM = 5.75*Pi/180.0;            // measured value from experiment
const double polarity = 1.0;                  //  A>0 is 1.0 ; A<0 is -1.0
const double A = 3.4;                        // units  nT AU^2, to have B = 5 nT at Earth (1AU, theta=90)
const double konvF = 9.0e-5/2.0;
const double SPbins[30] = { 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};


void TwoDimensionBpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
{
	srand(time(NULL));
	std::string destination = singleTone->getString("destination", "");
	if (destination.empty())
	{
		destination = getDirectoryName(singleTone);
	}
	if (!createDirectory("2DBP", destination))
	{
		return;
	}

	FILE *file = fopen("log.dat", "w");
	unsigned int nthreads = std::thread::hardware_concurrency();
	int new_MMM = ceil(12 / nthreads) * singleTone->getInt("millions", 1);

	for (int mmm = 0; mmm < new_MMM; mmm++)
	{
		std::vector<std::thread> threads;
		for (int i = 0; i < (int)nthreads; i++)
		{
			threads.emplace_back(std::thread(&TwoDimensionBpCpuSimulation::simulation, this));
		}
		for (auto &th : threads)
		{
			th.join();
		}
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
	double Tkinw, p, rp, dp, pp, pinj, cfactor, sumac;
	int m, mm;
	double dtheta, dKttkon, dTkin, thetap;
	double DriftR,arg,alphaH,Larmor,Bfield,f,fprime,DriftSheetR;
	thread_local std::random_device rd{};
	thread_local std::mt19937 generator(rd());
	thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (m = 0; m < 30; m++)
	{
		for (mm = 0; mm < 10000; mm++)
		{

			Tkininj = SPbins[m];
			Tkin = Tkininj;

			Tkinw = Tkin * 1e9 * q;						 /*v Jouloch*/
			Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q; /*vo Voltoch*/
			p = Rig * q / c;							 /*kg m s-1*/
			pinj = p;

			w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
			w = pow(w, -1.85) / p;
			w = w / 1e45;

			tt = Tkin + T0;
			t2 = tt + T0;
			beta = sqrt(Tkin * t2) / tt;

			r = 1.0;
			theta = Pi/2.;
			thetainj = theta;

			while (r < 100.0002)
			{ /* one history */
				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin * t2) / tt;

				alfa = t2/tt;
				
				Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

				r2 = r * r;

				gammma = (r * omega) * sin(theta) / V; // ZMENA  ; 1 chýba sin(theta)
       			gamma2 = gammma*gammma; // ZMENA
       			tmp1 = 1 + gamma2; // ZMENA
       			tem2 = tmp1 * tmp1;

				Bfactor = (5.0/3.4) *  r2 / sqrt(tmp1);        // SOLARPROP


				Kpar = K0*beta*Rig*Bfactor/3.0;
				if (Rig<0.1)
				{
					Kpar = K0*beta*0.1*Bfactor/3.0;
				}
				Kper = ratio * Kpar;   // SOLARPROP

				Krr = Kper + ((Kpar - Kper)/tmp1);
				Ktt = Kper;

				dtem1 = 2.0*r*omega*omega*sin(theta)*sin(theta)/(V*V);
				dKrr = ratio*K0*beta*Rig*((2.0*r*sqrt(tmp1)) - (r2*dtem1/(2.0*sqrt(tmp1))))/tmp1; // par Kperp ober par r
				dKrr = dKrr +  ((1.0-ratio)*K0*beta*Rig*((2.0*r*pow(tmp1,1.5))-(r2*dtem1*3.0*sqrt(tmp1)/2.0))/pow(tmp1,3.0));
				dKrr = dKrr*5.0/(3.0*3.4);     // SOLARPROP
				
				dKtt = (-1.0*ratio*K0*beta*Rig*r2*sin(theta)*cos(theta)*(omega*omega*r2/(V*V)))/pow(tmp1,1.5);

				dr = ((-1.0*V) + (2.0*Krr/r) + dKrr)*dt; // ZMENA - prva verzia je dKrr = 0
				dr = dr + (distribution(generator)*sqrt(2.0*Krr*dt));

				dtheta = (Ktt*cos(theta)) / (r2 * sin(theta));
				dtheta = dtheta*dt/tmp1;
				dtheta = dtheta +  ((distribution(generator)*sqrt(2.0*Ktt*dt)) / r); // ZMENA 8.11.2020

				dKttkon = (Ktt*cos(theta)) / (r2 * sin(theta));
				dKttkon = dKttkon +  (dKtt/r2); 

				dKttkon = (Ktt*cos(theta)) / (r2 * sin(theta));
				dKttkon = dKttkon/tmp1;

				dTkin = -2.0*V*alfa*Tkin*dt/(3.0*r);  
				
				Bfield = A*sqrt(tmp1)/(r*r);    // Parker field in nanoTesla, because A is in nanotesla

				Larmor = 0.0225*Rig/Bfield;     // SOLARPROP, maly ROZDIEL, PRECO?

				alphaH = Pi / sin(alphaM + (2.0*Larmor*Pi/(r*180.0)));   // PREVERIT v Burgerovom clanku 
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

				dr = dr - ((DriftR + DriftSheetR)*dt);

				rp = r; // Mirroring zapamatavanie
				pp = p;
				thetap = theta;  // ZMENA PB

				r = r + dr;
				theta = theta + dtheta;   //  ZMENA PB

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

				tt = Tkin + T0;
				t2 = tt + T0;
				beta = sqrt(Tkin*t2)/tt;

				if (beta > 0.001 && Tkin < 200.0 && r > 100.0)
				{
					outputMutex.lock();
					outputQueue.push({Tkininj,Tkin,r,w,thetainj,theta});
					outputMutex.unlock();
					break;
				}

				if (beta < 0.001)
				{
					break;
				}
				if (r < 0.1)
				{
					r = rp;
					p = pp;
					theta = thetap;
				}
			} 
		}	  
	}		  
}