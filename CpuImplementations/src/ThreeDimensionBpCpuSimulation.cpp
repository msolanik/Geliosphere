#include "ThreeDimensionBpCpuSimulation.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>

void ThreeDimensionBpCpuSimulation::runSimulation(ParamsCarrier *singleTone)
{
    singleTone->putFloat("K0", 5.0 * 1.43831e-5);
    spdlog::info("Starting initialization of 3D B-p simulation.");
    srand(time(NULL));
    std::string destination = singleTone->getString("destination", "");
    if (destination.empty())
    {
        destination = getDirectoryName(singleTone);
        spdlog::info("Destination is not specified - using generated name for destination: " + destination);
    }
    if (!createDirectory("3DBP", destination))
    {
        spdlog::error("Directory for 3D B-p simulations cannot be created.");
        return;
    }

    FILE *file = fopen("log.dat", "w");
    unsigned int nthreads = std::thread::hardware_concurrency();
    int new_MMM = ceil(singleTone->getInt("millions", 1) * 1000000 / (nthreads * 30 * 10000));
    setContants(singleTone, false);
    for (int mmm = 0; mmm < new_MMM; mmm++)
    {
        spdlog::info("Processed: {:03.2f}%", (float)mmm / ((float)new_MMM / 100.0));
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)nthreads; i++)
        {
            threads.emplace_back(std::thread(&ThreeDimensionBpCpuSimulation::simulation, this));
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

void ThreeDimensionBpCpuSimulation::simulation()
{
    double r, K, dr, arnum, theta, thetainj, Kpar, Bfactor, dtem1;
    double Tkin, Tkininj, Rig, tt, t2, beta, alfa, Ktt, dKrr;
    double w, r2, gammma, gamma2, tmp1, tem2, Kper, Krr, dKtt;
    double Tkinw, p, rp, dp, pp, pinj, cfactor, sumac;
    int m, mm;
    double dtheta, dKttkon, dTkin, thetap;
    double delta, deltarh, deltarh2;
    double DriftR, DriftTheta, arg, alphaH, Larmor, Bfield, f, fprime, DriftSheetR;
    double Kphph, Krt, Krph, Ktph, B111, B11, B12, B13, B22, B23;
    double sin2,sin3,dKtt0,dKtt1,dKtt2;
    double dKrtr, dKrtt;
    thread_local std::random_device rd{};
    thread_local std::mt19937 generator(rd());
    thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (m = 0; m < 30; m++)
    {
        for (mm = 0; mm < 10000; mm++)
        {
            Tkininj = SPbins[m];
            Tkin = Tkininj;

            Tkinw = Tkin * 1e9 * q;                      /*v Jouloch*/
            Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q; /*vo Voltoch*/
            p = Rig * q / c;                             /*kg m s-1*/
            pinj = p;

            w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
            w = pow(w, -1.85) / p;
            w = w / 1e45;

            tt = Tkin + T0;
            t2 = tt + T0;
            beta = sqrt(Tkin * t2) / tt;

            r = 1.0;

            theta = Pi / 2.;
            thetainj = theta;
            while (r < 100.0)
            { /* one history */

                tt = Tkin + T0;
                t2 = tt + T0;
                beta = sqrt(Tkin * t2) / tt;
                alfa = t2 / tt;
                Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));
                r2 = r * r;

                if ((theta < (1.7 * Pi / 180.0)) || (theta > (178.3 * Pi / 180.0)))
                {
                    delta = 0.2*0.003;
                }
                else
                {
                    delta = delta0 / sin(theta);
                }
                deltarh = delta / rh;
                deltarh2 = deltarh * deltarh;

                gammma = (r * omega) * sin(theta) / V; // ZMENA  ; 1 chýba sin(theta)
                gamma2 = gammma * gammma;              // ZMENA
                tmp1 = 1.0 + gamma2 + (r2 * deltarh2);   // NEW 062022
                tem2 = tmp1 * tmp1;
                Bfactor = (5. / 3.4) * r2 / sqrt(tmp1); // SOLARPROP

                if (Rig < 0.1)
                {
                    Kpar = K0 * beta * 0.1 * Bfactor / 3.0;
                }
                else 
                {
                    Kpar = K0 * beta * Rig * Bfactor / 3.0;
                }

                Kper = ratio * Kpar; // SOLARPROP

                Krr = Kper + ((Kpar - Kper) / tmp1);
                Ktt = Kper + (r2 * deltarh2 * (Kpar - Kper) / tmp1);
                // Kphph = (Kpar - Kper) * gamma2 / tmp1;
                Kphph = 1.0;

                Krt = deltarh * (Kpar - Kper) * r / tmp1;
                // Krph = (Kpar - Kper) * gammma / tmp1;
                Krph = 0.0;
                // Ktph = (Kpar - Kper) * r * deltarh * gammma / tmp1;
                Ktph = 0.0;

                B111 = (Kphph * Krt * Krt) - (2.0 * Krph * Krt * Ktph) + (Krr * Ktph * Ktph) + (Ktt * Krph * Krph) - (Krr * Ktt * Kphph);
                B11 = 2.0 * B111 / ((Ktph * Ktph) - (Ktt * Kphph));
                B11 = sqrt(B11);
                B12 = ((Krph * Ktph) - (Krt * Kphph)) / ((Ktph * Ktph) - (Ktt * Kphph));
                B12 = B12 * sqrt(2.0 * (Ktt - (Ktph * Ktph / Kphph)));
                B13 = sqrt(2.0) * Krph / sqrt(Kphph);
                B22 = Ktt - (Ktph * Ktph / Kphph);
                B22 = sqrt(2.0 * B22) / r;
                B23 = Ktph * sqrt(2.0 / Kphph) / r;

                dtem1 = 2.0 * r * omega * omega * sin(theta) * sin(theta) / (V * V);
                dtem1 = dtem1 + (2.0 * r * deltarh2);                                                                 // NEW 062022
                dKrr = ratio * K0 * beta * Rig * ((2.0 * r * sqrt(tmp1)) - (r2 * dtem1 / (2.0 * sqrt(tmp1)))) / tmp1; // par Kperp ober par r
                dKrr = dKrr + ((1.0 - ratio) * K0 * beta * Rig * ((2.0 * r * pow(tmp1, 1.5)) - (r2 * dtem1 * 3.0 * sqrt(tmp1) / 2.0)) / pow(tmp1, 3.0));
                dKrr = dKrr * 5. / (3. * 3.4); // SOLARPROP

                sin3 = sin(theta)*sin(theta)*sin(theta);

                dKtt = sin(theta)*cos(theta)*(omega*omega*r2/(V*V));
                dKtt = dKtt - (r2*delta0*delta0*cos(theta)/(rh*rh*sin3));
                dKtt = (-1.0*ratio*K0*beta*Rig*r2*dKtt)/pow(tmp1,1.5);

                if ((theta>(1.7*Pi/180.))&&(theta<(178.3*Pi/180.0))) 
                {
                    dKtt0 = 3.0*(1.0-ratio)*K0*beta*Rig*r2*r2*deltarh2;
                    dKtt1 = omega*omega*r2*sin(theta)*cos(theta)/(V*V*pow(tmp1,2.5));
                    dKtt = dKtt - (dKtt0*dKtt1);
                }
                else
                {
                    sin2 = sin(theta)*sin(theta);
                    dKtt0 = (1.0-ratio)*K0*beta*Rig*r2*r2*delta0*delta0/(rh*rh);
                    dKtt1 = -2.0*(cos(theta)/sin3)/pow(tmp1,1.5);
                    dKtt2 = 1.5*((2.0*omega*omega*r2*sin(theta)*cos(theta)/(V*V)) - (2.0*r2*delta0*delta0*cos(theta)/(rh*rh*sin3)))/(sin2*pow(tmp1,2.5));
                    dKtt = dKtt + (dKtt0*(dKtt1 - dKtt2));
                }

                dKrtr = (1.0 - ratio) * K0 * beta * Rig * deltarh * 3.0 * r2 / pow(tmp1, 2.5);

                if ((theta > (1.7 * Pi / 180.)) && (theta < (178.3 * Pi / 180.0)))
                {
                    dKrtt = (1.0 - ratio) * K0 * beta * Rig * r2 * r / (rh * pow(tmp1, 2.5));
                    dKrtt = -1.0 * dKrtt * delta;
                    dKrtt = dKrtt * 3.0 * gamma2 * cos(theta) / sin(theta);
                }
                else
                {
                    dKrtt = (1.0 - ratio) * K0 * beta * Rig * r2 * r / (rh * pow(tmp1, 2.5));
                    dKrtt = -1.0 * dKrtt * delta0 * cos(theta) / (sin(theta) * sin(theta));
                    dKrtt = dKrtt*(1.0 - (2.0*r2*deltarh2) + (4.0*gamma2)); 
                }

                dr = ((-1.0 * V) + (2.0 * Krr / r) + dKrr) * dt;                  
                dr = dr + (dKrtt * dt / r) + (Krt * cos(theta) * dt / (r * sin(theta))); 
                dr = dr + (distribution(generator) * B11 * sqrt(dt));
                dr = dr + (distribution(generator) * B12 * sqrt(dt));
                dr = dr + (distribution(generator) * B13 * sqrt(dt));

                dtheta = (Ktt * cos(theta)) / (r2 * sin(theta));
                dtheta = (dtheta*dt) + (dKtt*dt/r2);
                dtheta = dtheta + (dKrtr * dt) + (2.0 * Krt * dt / r);                                                     // NEW 062022
                dtheta = dtheta + (distribution(generator) * B22 * sqrt(dt)) + (distribution(generator) * B23 * sqrt(dt)); // NEW 06@)@@

                // SOLARPROP
                dKttkon = (Ktt * cos(theta)) / (r2 * sin(theta));
                dKttkon = dKttkon / tmp1;

                dTkin = -2.0 * V * alfa * Tkin * dt / (3.0 * r);

                // Bfield = A * sqrt(tmp1) / (r * r); // Parker field in nanoTesla, because A is in nanotesla
                // Larmor = 0.0225 * Rig / Bfield;    // SOLARPROP, maly ROZDIEL, PRECO?

                // alphaH = Pi / sin(alphaM + (2.0 * Larmor * Pi / (r * 180.0))); // PREVERIT v Burgerovom clanku
                // alphaH = alphaH - 1.0;
                // alphaH = 1.0 / alphaH;
                // alphaH = acos(alphaH);

                // arg = (1. - (2. * theta / Pi)) * tan(alphaH);
                // f = atan(arg) / alphaH;

                // DriftR = polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * cos(theta) * gammma * f / (tem2 * sin(theta));
                // DriftTheta = -1.0 * polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * gammma * (2.0 + (gammma * gammma)) * f / tem2;
                // fprime = 1.0 + (arg * arg);
                // fprime = tan(alphaH) / fprime;
                // fprime = -1.0 * fprime * 2.0 / (Pi * alphaH);

                // DriftSheetR = polarity * konvF * (1.0 / (3.0 * A)) * Rig * beta * r * gammma * fprime / tmp1;
                // dr = dr + ((DriftR + DriftSheetR) * dt);
                // dtheta = dtheta + (DriftTheta * dt / r);
                r = r + dr;
                Tkin = Tkin - dTkin; 
                theta = theta + dtheta;
                if (theta < 0.0)
                {
                    theta = fabs(theta);
                }
                if (theta > 2.0 * Pi)
                {
                    theta = theta - (2.0 * Pi);
                }
                else if (theta > Pi)
                {
                    theta = (2.0 * Pi) - theta;
                }

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