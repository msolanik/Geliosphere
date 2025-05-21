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

    std::vector<SimulationOutput> localOutputs;
    localOutputs.reserve(30 * 500);

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
            p = GetMomentum(Rig);
            w = W(p);

            r = rInit;
            theta = thetainj;

            while (r < 100.0)
            {
                beta = Beta(Tkin);
                Rig = RigFromTkin(Tkin);

                r2 = r * r;

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

                gamma = (r * omega) * sin(theta) / V;
                gamma2 = gamma * gamma;

                Cb = 1.0 + gamma2 + (r2 * deltarh2);
                Cb2 = Cb * Cb;
                Bfactor = (5. / 3.4) * r2 / sqrt(Cb);

                if (Rig < 0.1)
                {
                    Kpar = K0 * beta * 0.1 * Bfactor / 3.0;
                }
                else
                {
                    Kpar = K0 * beta * Rig * Bfactor / 3.0;
                }

                Kper = ratio * Kpar;

                Krr = Kper + ((Kpar - Kper) / Cb);
                Ktt = Kper + (r2 * deltarh2 * (Kpar - Kper) / Cb);
                Kphph = 1.0;

                Krt = deltarh * (Kpar - Kper) * r / Cb;
                Krph = 0.0;
                Ktph = 0.0;

                B11Temp = (Kphph * Krt * Krt) - (2.0 * Krph * Krt * Ktph) + (Krr * Ktph * Ktph) + (Ktt * Krph * Krph) - (Krr * Ktt * Kphph);
                B11 = 2.0 * B11Temp / ((Ktph * Ktph) - (Ktt * Kphph));
                B11 = sqrt(B11);

                B12 = ((Krph * Ktph) - (Krt * Kphph)) / ((Ktph * Ktph) - (Ktt * Kphph));
                B12 = B12 * sqrt(2.0 * (Ktt - (Ktph * Ktph / Kphph)));

                B13 = sqrt(2.0) * Krph / sqrt(Kphph);

                B22 = Ktt - (Ktph * Ktph / Kphph);
                B22 = sqrt(2.0 * B22) / r;

                B23 = Ktph * sqrt(2.0 / Kphph) / r;

                COmega = 2.0 * r * omega * omega * sin(theta) * sin(theta) / (V * V);
                COmega = COmega + (2.0 * r * deltarh2);

                dKper = ratio * K0 * beta * Rig * ((2.0 * r * sqrt(Cb)) - (r2 * COmega / (2.0 * sqrt(Cb)))) / (3.0 * (5.0 / 3.4) * Cb);

                dKrr = dKper + ((1.0 - ratio) * K0 * beta * Rig * ((2.0 * r * pow(Cb, 1.5)) - (r2 * COmega * 3.0 * sqrt(Cb) / 2.0)) / (3.0 * (5.0 / 3.4) * pow(Cb, 3.0)));

                if ((theta > (1.7 * Pi / 180.)) && (theta < (178.3 * Pi / 180.0)))
                {
                    CKtt = sin(theta) * cos(theta) * (omega * omega * r2 / (V * V));
                    dKtt1 = (-1.0 * ratio * K0 * beta * Rig * r2 * CKtt) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));
                    dKtt2 = (1.0 - ratio) * K0 * beta * Rig * r2 * r2 * deltarh2;
                    dKtt4 = 3.0 * CKtt / pow(Cb, 2.5);
                    dKtt = dKtt1 - (dKtt2 * dKtt4);
                }
                else
                {
                    sin2 = sin(theta) * sin(theta);
                    sin3 = sin(theta) * sin(theta) * sin(theta);

                    CKtt = sin(theta) * cos(theta) * (omega * omega * r2 / (V * V));
                    CKtt = CKtt - (r2 * delta0 * delta0 * cos(theta) / (rh * rh * sin3));
                    dKtt1 = (-1.0 * ratio * K0 * beta * Rig * r2 * CKtt) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));
                    dKtt2 = (1.0 - ratio) * K0 * beta * Rig * r2 * r2 * delta0 * delta0 / (rh * rh);
                    dKtt3 = -2.0 * (cos(theta) / sin3) / (3.0 * (5.0 / 3.4) * pow(Cb, 1.5));
                    dKtt4 = 3.0 * (CKtt / sin2) / (3.0 * (5.0 / 3.4) * pow(Cb, 2.5));
                    dKtt = dKtt1 + (dKtt2 * (dKtt3 - dKtt4));
                }

                dKrtr = (1.0 - ratio) * K0 * beta * Rig * deltarh * r2 / (3.0 * (5.0 / 3.4) * pow(Cb, 2.5));

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

                dr = ((-1.0 * V) + (2.0 * Krr / r) + dKrr) * dt;
                dr = dr + (dKrtt * dt / r) + (Krt * cos(theta) * dt / (r * sin(theta)));
                dr = dr + (distribution(generator) * B11 * sqrt(dt));
                dr = dr + (distribution(generator) * B12 * sqrt(dt));
                dr = dr + (distribution(generator) * B13 * sqrt(dt));

                dtheta = (Ktt * cos(theta)) / (r2 * sin(theta));
                dtheta = (dtheta * dt) + (dKtt * dt / r2);
                dtheta = dtheta + (dKrtr * dt) + (2.0 * Krt * dt / r);
                dtheta = dtheta + (distribution(generator) * B22 * sqrt(dt)) + (distribution(generator) * B23 * sqrt(dt));

                alfa = (Tkin + T0 + T0)/(Tkin + T0);
                dTkin = -2.0 * V * alfa * Tkin * dt / (3.0 * r);

                Bfield = A * sqrt(Cb) / (r * r);
                Larmor = LarmorRadius(Rig, Bfield);
                alphaH = AlphaH(Larmor, r);
                f = ComputeF(theta, alphaH);
                fprime = ComputeFPrime(theta, alphaH);

                DriftR = polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * cos(theta) * gamma * f / (Cb2 * sin(theta));
                DriftTheta = -1.0 * polarity * konvF * (2.0 / (3.0 * A)) * Rig * beta * r * gamma * (2.0 + (gamma * gamma)) * f / Cb2;
                DriftSheetR = polarity * konvF * (1.0 / (3.0 * A)) * Rig * beta * r * gamma * fprime / Cb;

                dr = dr + ((DriftR + DriftSheetR) * dt);
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
                    
                    localOutputs.emplace_back(SimulationOutput{Tkininj, Tkin, r, w, thetainj, theta});
                    
                    break;
                }
            }
        }
    }
    std::lock_guard<std::mutex> lock(outputMutex);
	for (const auto& output : localOutputs)
	{
		outputQueue.push(output);
	}
}




double GeliosphereCpuModel::Beta(double Tkin) {
    return sqrt(Tkin * (Tkin + 2.0 * T0)) / (Tkin + T0);
}

double GeliosphereCpuModel::RigFromTkin(double Tkin) {
    return sqrt(Tkin * (Tkin + 2.0 * T0));
}

double GeliosphereCpuModel::W(double p) {
    double w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
    return pow(w, -1.85) / p / 1e45;
}

double GeliosphereCpuModel::GetMomentum(double Rig) {
    return Rig * q / c;
}

double GeliosphereCpuModel::LarmorRadius(double Rig, double Bfield) {
    return 0.0225 * Rig / Bfield;
}

double GeliosphereCpuModel::AlphaH(double Larmor, double r) {
    double alphaH = Pi / sin(alphaM + (2.0 * Larmor * Pi / (r * 180.0)));
    alphaH = 1.0 / (alphaH - 1.0);
    return acos(alphaH);
}

double GeliosphereCpuModel::ComputeF(double theta, double alphaH) {
    double arg = (1. - (2. * theta / Pi)) * tan(alphaH);
    return atan(arg) / alphaH;
}

double GeliosphereCpuModel::ComputeFPrime(double theta, double alphaH) {
    double arg = (1. - (2. * theta / Pi)) * tan(alphaH);
    double fprime = tan(alphaH) / (1.0 + arg * arg);
    return -1.0 * fprime * 2.0 / (Pi * alphaH);
}