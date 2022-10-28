#include "TwoDimensionBpResults.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "spdlog/spdlog.h"

#include "ResultsUtils.hpp"
#include "ResultConstants.hpp"

void TwoDimensionBpResults::runAlgorithm(ParamsCarrier *singleTone)
{
    spdlog::info("Started to analyze 2D/3D B-p particles.");
    ResultsUtils *resultsUtils = new ResultsUtils();
    double w, Rig, p1AU, Tkin, r, p, Tkinw, Rig1AU, Tkininj, theta, thetainj, tt, t2, beta;
    double tem6, tem5, jlis, tem, wJGR;
    double speSP[31] = {0}, speJGR[31] = {0}, speN[31] = {0};
    FILE *inputFile = fopen("log.dat", "r");
    int numberOfIterations = resultsUtils->countLines(inputFile) - 1;
    int targetArray[] = {numberOfIterations};
    spdlog::info("Founded {} to analyze.", numberOfIterations);
    for (int i = 0; i < numberOfIterations; i++)
    {
        int reader = fscanf(inputFile, "%lf %lf %lf %lf %lf %lf\n", &Tkininj, &Tkin, &r, &w, &thetainj, &theta);
        if (reader == -1)
        {
            spdlog::error("Could not read from log.dat file.");
            return;
        }
        Tkinw = Tkin * 1e9 * q;
        Rig = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
        p = Rig * q / c;
        Tkinw = Tkininj * 1e9 * q;
        Rig1AU = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
        p1AU = Rig1AU * q / c;

        tem5 = 21.1 * exp(-2.8 * log(Tkininj));
        tem6 = 1.0 + (5.85 * exp(-1.22 * log(Tkininj))) + (1.18 * exp(-2.54 * log(Tkininj)));
        jlis = tem5 / tem6;
        w = jlis / (p * p);
        w = w * p1AU * p1AU;

        tt = Tkin + Tr;
        t2 = tt + Tr;
        beta = sqrt(Tkin * t2) / tt;
        tem = (Tkin + 0.67) / 1.67;
        tem = exp(-3.93 * log(tem));
        jlis = 2.7e3 * exp(1.12 * log(Tkin)) * tem;
        jlis = jlis / (beta * beta);

        wJGR = jlis / (p * p);
        wJGR = wJGR * p1AU * p1AU;
        for (int ii = 0; ii < 30; ii++)
        {
            if ((Tkininj > SPbins[ii] * 0.99) && (Tkininj < SPbins[ii] * 1.01))
            {
                speSP[ii + 1] = speSP[ii + 1] + w;
                speJGR[ii + 1] = speJGR[ii + 1] + wJGR;
                speN[ii + 1]++;
            }
        }
    }
    fclose(inputFile);

    struct spectrumOutput *spectrumOutput;
    spdlog::info("1.");
    spectrumOutput->fileName = std::string("JGAR");
    spdlog::info("2.");
    spectrumOutput->isCsv = singleTone->getInt("csv", 0);
    spdlog::info("3.");
    resultsUtils->writeSpectrum(spectrumOutput, speN, speJGR, SPECTRUM_SOLARPROP);
    spdlog::info("Spectrum based on JGAR has been written to file.");

    spectrumOutput->fileName = std::string("Weber");
    spectrumOutput->isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(spectrumOutput, speN, speSP, SPECTRUM_SOLARPROP);
    spdlog::info("Spectrum based on Weber has been written to file.");
}