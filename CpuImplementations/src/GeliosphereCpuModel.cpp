#include "GeliosphereCpuModel.hpp"
#include "FileUtils.hpp"
#include "Constants.hpp"

#include "spdlog/spdlog.h"

#include <thread>
#include <random>
#include <cmath>

void GeliosphereCpuModel::runSimulation(ParamsCarrier *singleTone)
{
    spdlog::info("Starting initialization of simulation for Geliosphere 2D model.");
    srand(time(NULL));
    std::string destination = singleTone->getString("destination", "");
    if (destination.empty())
    {
        destination = getDirectoryName(singleTone);
        spdlog::info("Destination is not specified - using generated name for destination: " + destination);
    }
    if (!createDirectory("Geliosphere2D", destination))
    {
        spdlog::error("Directory for Geliosphere 2D simulations cannot be created.");
        return;
    }

    FILE *file = fopen("log.dat", "w");
    unsigned int nthreads = std::thread::hardware_concurrency();
    int targetIterations = ceil((double)singleTone->getInt("millions", 1) * 1000000.0 / ((double)nthreads * 30.0 * 500.0));
    setContants(singleTone);
    for (int iteration = 0; iteration < targetIterations; iteration++)
    {
        spdlog::info("Processed: {:03.2f}%", (float) iteration / ((float)targetIterations / 100.0));
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)nthreads; i++)
        {
            threads.emplace_back(std::thread(&GeliosphereCpuModel::simulation, this, i, nthreads, iteration));
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
    writeSimulationReportFile(singleTone);
}

void GeliosphereCpuModel::simulation(int threadNumber, unsigned int availableThreads, int iteration)
{
    double r, dr, arnum, theta, Kpar, Bfactor, COmega, Tkinp;
    double Tkin, Tkininj, Rig, beta, alfa, Ktt, dKrr, dKper;
    double w, r2, gamma, gamma2, Cb, Cb2, Kper, Krr, dKtt, dKtt3, dKtt4, CKtt;
    double Tkinw, p;
    int m, mm;
    double dtheta, dTkin;
    double delta, deltarh, deltarh2;
    double DriftR, DriftTheta, arg, alphaH, Larmor, Bfield, f, fprime, DriftSheetR;
    double Kphph, Krt, Krph, Ktph, B11Temp, B11, B12, B13, B22, B23;
    double sin2, sin3, dKtt1, dKtt2;
    double dKrtr, dKrtt;
    thread_local std::random_device rd{};
    thread_local std::mt19937 generator(rd());
    thread_local std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (int energy = 0; energy < 30; energy++)
    {
        for (int particlePerEnergy = 0; particlePerEnergy < 500; particlePerEnergy++)
        {
			Tkininj = (useUniformInjection) 
				? getTkinInjection(((availableThreads * iteration + threadNumber) * 500) + particlePerEnergy, 0.0001, uniformEnergyInjectionMaximum, 10000)
				: SPbins[energy];
			Tkin = Tkininj;

            Tkinw = Tkin * 1e9 * q;
            Rig = sqrt(Tkinw * (Tkinw + (2.0 * T0w))) / q;
            p = Rig * q / c;

			// Equation under Equation 6 from 
            // Yamada et al. "A stochastic view of the solar modulation phenomena of cosmic rays" GEOPHYSICAL RESEARCH LETTERS, VOL. 25, NO.13, PAGES 2353-2356, JULY 1, 1998
            // https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/98GL51869
            w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
            w = pow(w, -1.85) / p;
            w = w / 1e45;

            r = rInit;
            theta = thetainj;

            while (r < 100.0)
            {
				// Equation 5
                // Link to Equation 5 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/1D_models_description.ipynb#5
				beta = sqrtf(Tkin * (Tkin + T0 + T0)) / (Tkin + T0);

                // Equation 8 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf		
                Rig = sqrt(Tkin * (Tkin + (2.0 * T0)));

                r2 = r * r;

                // Equation 25
                // Link to Equation 25 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#25
                if (theta < (1.7 * Pi / 180.) || theta > (178.3 * Pi / 180.0))
                {
                    delta = 0.003;
                }
                else
                {
                    delta = delta0 / sin(theta);
                }

                deltarh = delta / rh;
                deltarh2 = deltarh * deltarh;

                // Equation 24
                // Link to Equation 24 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#24
                gamma = (r * omega) * sin(theta) / V;
                gamma2 = gamma * gamma;

                // Equation 26
                // Link to Equation 26 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#26
                Cb = 1.0 + gamma2 + (r2 * deltarh2);
                Cb2 = Cb * Cb;
                Bfactor = (5. / 3.4) * r2 / sqrt(Cb);

                // Equation 32  
                // Link to Equation 32 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#32
                if (Rig < 0.1)
                {
                    Kpar = K0 * beta * 0.1 * Bfactor / 3.0;
                }
                else
                {
                    Kpar = K0 * beta * Rig * Bfactor / 3.0;
                }

                // Equation 33
                // Link to Equation 33 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#33
                Kper = ratio * Kpar;

                // Equation 27
                // Link to Equation 27 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#27
                Krr = Kper + ((Kpar - Kper) / Cb);
                
                // Equation 28
                // Link to Equation 28 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#28
                Ktt = Kper + (r2 * deltarh2 * (Kpar - Kper) / Cb);
                Kphph = 1.0;

                // Equation 29
                // Link to Equation 29 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#29
                Krt = deltarh * (Kpar - Kper) * r / Cb;
                Krph = 0.0;
                Ktph = 0.0;

                // Equation 16, where Krph = Ktph = 0, and Kphph = 1 
                // Link to Equation 16 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#16
                B11Temp = (Kphph * Krt * Krt) - (2.0 * Krph * Krt * Ktph) + (Krr * Ktph * Ktph) + (Ktt * Krph * Krph) - (Krr * Ktt * Kphph);
                B11 = 2.0 * B11Temp / ((Ktph * Ktph) - (Ktt * Kphph));
                B11 = sqrt(B11);

                // Equation 17
                // Link to Equation 17 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#17
                B12 = ((Krph * Ktph) - (Krt * Kphph)) / ((Ktph * Ktph) - (Ktt * Kphph));
                B12 = B12 * sqrt(2.0 * (Ktt - (Ktph * Ktph / Kphph)));

                // Equation 20
                // Link to Equation 20 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#20
                B13 = sqrt(2.0) * Krph / sqrt(Kphph);

                // Equation 18
                // Link to Equation 18 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#18
                B22 = Ktt - (Ktph * Ktph / Kphph);
                B22 = sqrt(2.0 * B22) / r;

                // Equation 20
                // Link to Equation 20 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#20
                B23 = Ktph * sqrt(2.0 / Kphph) / r;

                // Equation 34
                // Link to Equation 34 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#34
                COmega = 2.0 * r * omega * omega * sin(theta) * sin(theta) / (V * V);
                COmega = COmega + (2.0 * r * deltarh2);
                
                // Equation 36
                // Link to Equation 36 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#36
                dKper = ratio * K0 * beta * Rig * ((2.0 * r * sqrt(Cb)) - (r2 * COmega / (2.0 * sqrt(Cb)))) / (3.0 * (5.0 / 3.4) * Cb);

                // Equation 35 
                // Link to Equation 35 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#35               
                dKrr = dKper + ((1.0 - ratio) * K0 * beta * Rig * ((2.0 * r * pow(Cb, 1.5)) - (r2 * COmega * 3.0 * sqrt(Cb) / 2.0)) / ( 3.0 * (5.0 / 3.4) * pow(Cb, 3.0)));

                if ((theta > (1.7 * Pi / 180.)) && (theta < (178.3 * Pi / 180.0)))
                {
                    // Equation 37
                    // Link to Equation 37 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#37
                    CKtt = sin(theta) * cos(theta) * (omega * omega * r2 / (V * V));
                 
                    // Equation 38
                    // Link to Equation 38 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#38
                    dKtt1 = (-1.0 * ratio * K0 * beta * Rig * r2 * CKtt) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));

                    // Equation 39
                    // Link to Equation 39 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#39
                    dKtt2 = (1.0 - ratio) * K0 * beta * Rig * r2 * r2 * deltarh2;
                    
                    // Equation 41
                    // Link to Equation 41 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#41
                    dKtt4 = 3.0 * CKtt / pow(Cb, 2.5);
                    
                    // Equation 42
                    // Link to Equation 42 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#42
                    dKtt = dKtt1 - (dKtt2 * dKtt4);
                }
                else
                {
                    sin2 = sin(theta) * sin(theta);
                    sin3 = sin(theta) * sin(theta) * sin(theta);

                    // Equation 37
                    // Link to Equation 37 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#37
                    CKtt = sin(theta) * cos(theta) * (omega * omega * r2 / (V * V));
                    CKtt = CKtt - (r2 * delta0 * delta0 * cos(theta) / (rh * rh * sin3));
                    
                    // Equation 38
                    // Link to Equation 38 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#38
                    dKtt1 = (-1.0 * ratio * K0 * beta * Rig * r2 * CKtt) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));

                    // Equation 39
                    // Link to Equation 39 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#39
                    dKtt2 = (1.0 - ratio) * K0 * beta * Rig * r2 * r2 * delta0 * delta0 / (rh * rh);
                    
                    // Equation 40
                    // Link to Equation 40 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#40
                    dKtt3 = -2.0 * (cos(theta) / sin3) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));
                    
                    // Equation 41
                    // Link to Equation 41 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#41
                    dKtt4 = 3.0 * (CKtt / sin2) / (3.0 * (5.0 / 3.4) * pow(Cb, 2.5));
                    
                    // Equation 42
                    // Link to Equation 42 in Jupyter Notebook Documentation: 
                    // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#42
                    dKtt = dKtt1 + (dKtt2 * (dKtt3 - dKtt4));
                }

                // Equation 43
                // Link to Equation 43 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#43
                dKrtr = (1.0 - ratio) * K0 * beta * Rig * deltarh * r2 / (3.0 * (5.0 / 3.4) * pow(Cb, 2.5));

                // Equation 44
                // Link to Equation 44 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#44
                if ((theta > (1.7 * Pi / 180.)) && (theta < (178.3 * Pi / 180.0)))
                {
                    dKrtt = (1.0 - ratio) * K0 * beta * Rig * r2 * r / ((3.0 * (5.0 / 3.4) * rh * pow(Cb, 2.5)));
                    dKrtt = -1.0 * dKrtt * delta;
                    dKrtt = dKrtt * 3.0 * gamma2 * cos(theta) / sin(theta);
                }
                else
                {
                    dKrtt = (1.0 - ratio) * K0 * beta * Rig * r2 * r / ((3.0 * (5.0 / 3.4) * rh * pow(Cb, 2.5)));
                    dKrtt = -1.0 * dKrtt * delta0 * cos(theta) / (sin(theta) * sin(theta));
                    dKrtt = dKrtt * (1.0 - (2.0 * r2 * deltarh2) + (4.0 * gamma2));
                }

                // Equation 21
                // Link to Equation 21 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#21
                dr = ((-1.0 * V) + (2.0 * Krr / r) + dKrr) * dt;
                dr = dr + (dKrtt * dt / r) + (Krt * cos(theta) * dt / (r * sin(theta)));
                dr = dr + (distribution(generator) * B11 * sqrt(dt));
                dr = dr + (distribution(generator) * B12 * sqrt(dt));
                dr = dr + (distribution(generator) * B13 * sqrt(dt));

                // Equation 22
                // Link to Equation 22 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#22
                dtheta = (Ktt * cos(theta)) / (r2 * sin(theta));
                dtheta = (dtheta * dt) + (dKtt * dt / r2);
                dtheta = dtheta + (dKrtr * dt) + (2.0 * Krt * dt / r);
                dtheta = dtheta + (distribution(generator) * B22 * sqrt(dt)) + (distribution(generator) * B23 * sqrt(dt));

                // Equation 23
                // Link to Equation 23 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#23
				alfa = (Tkin + T0 + T0)/(Tkin + T0);
                dTkin = -2.0 * V * alfa * Tkin * dt / (3.0 * r);

                Bfield = A * sqrt(Cb) / (r * r);
				
                // Equation 26 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                Larmor = 0.0225 * Rig / Bfield;

                // Equation 34 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                alphaH = Pi / sin(alphaM + (2.0 * Larmor * Pi / (r * 180.0)));
                alphaH = alphaH - 1.0;
                alphaH = 1.0 / alphaH;
                alphaH = acos(alphaH);

                // Equation 32 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                arg = (1. - (2. * theta / Pi)) * tan(alphaH);
                f = atan(arg) / alphaH;

                // Equation 35 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                DriftR = polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * cos(theta) * gamma * f / (Cb2 * sin(theta));
                
                // Equation 36 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                DriftTheta = -1.0 * polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * gamma * (2.0 + (gamma * gamma)) * f / Cb2;

                // Equation 33 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                fprime = 1.0 + (arg * arg);
                fprime = tan(alphaH) / fprime;
                fprime = -1.0 * fprime * 2.0 / (Pi * alphaH);
                
                // Equation 37 from 
                // Kappl, Rolf. "SOLARPROP: Charge-sign dependent solar modulation for everyone." Computer Physics Communications 207 (2016): 386-399.
                // https://arxiv.org/pdf/1511.07875.pdf
                DriftSheetR = polarity * konvF * (1.0 / (3.0 * A)) * Rig * beta * r * gamma * fprime / Cb;

                // Equation 21
                // Link to Equation 21 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#21
                dr = dr + ((DriftR + DriftSheetR) * dt);
                
                // Equation 22
                // Link to Equation 22 in Jupyter Notebook Documentation: 
                // https://nbviewer.org/github/msolanik/Geliosphere/blob/main/ModelDocs/geliosphere-model-description.ipynb#22
                dtheta = dtheta + (DriftTheta * dt / r);

                r = r + dr;
                theta = theta + dtheta;
                Tkin = Tkin - dTkin;

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
                    outputQueue.push({Tkininj, Tkin, r, w, thetainj, theta});
                    outputMutex.unlock();
                    break;
                }
            }
        }
    }
}