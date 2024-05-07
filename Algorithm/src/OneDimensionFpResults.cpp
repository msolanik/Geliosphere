#include "OneDimensionFpResults.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "spdlog/spdlog.h"

#include "ResultsUtils.hpp"
#include "ResultConstants.hpp"
#include <string>

void OneDimensionFpResults::runAlgorithm(ParamsCarrier *singleTone)
{
    spdlog::info("Started to analyze 1D F-p particles.");
    ResultsUtils *resultsUtils = new ResultsUtils();
    int energy1e2, energy1e3, energy4e2;
    double Rig, p1AU, Tkin, w0, w, p100AU, r, sumac;
    double binb[25], binw[25], binc[25];
    double spe1e2[100] = {0}, spe1e2N[100] = {0};
    double spe4e2[400] = {0}, spe4e2N[400] = {0};
    double spe1e3[1000] = {0}, spe1e3N[1000] = {0};
    double spelog[24] = {0};
    for (int i = 0; i < 25; i++)
    {
        binb[i] = exp((((i + 1) / 5.0) - 3.0) * log(10));
    }
    for (int i = 0; i < 24; i++)
    {
        binw[i] = binb[i + 1] - binb[i];
        binc[i] = (binb[i + 1] + binb[i]) / 2.0;
    }
    FILE *inputFile = fopen(singleTone->getString("pathToLogFile","log.dat").c_str(), "r");
    int numberOfIterations = resultsUtils->countLines(inputFile) - 1;
    if (numberOfIterations < 0)
    {
        spdlog::info("No trajectory found in log file.", numberOfIterations);
        spdlog::warn("Please, consider increase of amount of simulated test particles for current input parameters.", numberOfIterations);
    }
    else
    {
        spdlog::info("Founded {} trajectories for analysis.", numberOfIterations);
    }
    for (int i = 0; i < numberOfIterations; i++)
    {
        int reader = fscanf(inputFile, " %lf  %lf  %lf  %lf %lf \n", &p100AU, &p1AU, &r, &w0, &sumac);
        if (reader == -1)
        {
            spdlog::error("Could not read from log.dat file.");
            return;
        }
        Rig = p1AU * c / q;
        Tkin = sqrt((T0 * T0 * q * q * 1e9 * 1e9) + (q * q * Rig * Rig)) - (T0 * q * 1e9);
        Tkin = Tkin / (q * 1e9);
        w0 = (m0 * m0 * c * c * c * c) + (p100AU * p100AU * c * c);
        w0 = exp(-1.85 * log(w0)) / p100AU;
        w0 = w0 / 1e45;
        w = w0 * p1AU * p1AU * exp(sumac);
        if (r < 0.3)
        {
            w = 0.0;
        }
        for (int i = 0; i < 24; i++)
        {
            if (Tkin > binb[i] && Tkin < binb[i + 1])
            {
                spelog[i] += w;
            }
        }
        energy1e2 = (int)trunc(Tkin);
        if (energy1e2 > 0 && energy1e2 < 101)
        {
            spe1e2[energy1e2 - 1] += w;
            spe1e2N[energy1e2 - 1] += 1;
        }
        energy1e3 = (int)trunc((Tkin + 0.05) * 10);
        if (energy1e3 > 0 && energy1e3 < 1001)
        {
            spe1e3[energy1e3 - 1] += w;
            spe1e3N[energy1e3 - 1] += 1;
        }
        energy4e2 = (int)trunc((Tkin + 0.125) * 4);
        if (energy4e2 > 0 && energy4e2 < 401)
        {
            spe4e2[energy4e2 - 1] += w;
            spe4e2N[energy4e2 - 1] += 1;
        }
    }
    fclose(inputFile);

    struct spectrumOutput spectrumOutput;
    spectrumOutput.fileName = "output_1e3bin";
    spectrumOutput.size = 1000;
    spectrumOutput.tkinPortion = 10;
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, spe1e3N, spe1e3, SPECTRUM_1E3);
    spdlog::info("Spectrum has been written to file containing 1e3 bins.");

    spectrumOutput.fileName = "output_1e2bin";
    spectrumOutput.size = 100;
    spectrumOutput.tkinPortion = 1;
    resultsUtils->writeSpectrum(&spectrumOutput, spe1e2N, spe1e2, SPECTRUM_1E2);
    spdlog::info("Spectrum has been written to file containing 1e2 bins.");

    spectrumOutput.fileName = "output_logbin";
    resultsUtils->writeSpectrum(&spectrumOutput, NULL, spelog, SPECTRUM_LOG);
    spdlog::info("Spectrum has been written to file containing logarithmic bins.");

    spectrumOutput.fileName = "output_4e2bin";
    spectrumOutput.size = 400;
    spectrumOutput.tkinPortion = 4;
    resultsUtils->writeSpectrum(&spectrumOutput, spe4e2N, spe4e2, SPECTRUM_4E2);
    spdlog::info("Spectrum has been written to file containing 4e2 bins.");
}