#include "TwoDimensionBpResults.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "spdlog/spdlog.h"

#include "ResultsUtils.hpp"
#include "ResultConstants.hpp"
#include <string>

void TwoDimensionBpResults::runAlgorithm(ParamsCarrier *singleTone)
{
    spdlog::info("Started to analyze 2D B-p particles.");
    ResultsUtils *resultsUtils = new ResultsUtils();
    double w, Rig, p1AU, Tkin, r, p, Tkinw, Rig1AU, Tkininj, theta, thetainj, tt, t2, beta;
    double tem6, tem5, jlis, tem, wJGR, jlisJGR, jlisBurger, wBurger;
    double speSP[31] = {0}, speJGR[31] = {0}, speN[31] = {0}, speBurger[31] = {0}, ulyssesBin[4] = {0}, ulyssesBinN[4] = {0}, amsBin[72] = {0}, amsBinN[72] = {0};
    FILE *inputFile = fopen(singleTone->getString("pathToLogFile","log.dat").c_str(), "r");
    int numberOfIterations = resultsUtils->countLines(inputFile) - 1;
    int targetArray[] = {numberOfIterations};
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
        
        // Based on:
        // W. R. Webber, P. R. Higbie. "Production of cosmogenic Be nuclei in the Earth's atmosphere by cosmic rays: Its dependence on solar modulation and the interstellar cosmic ray spectrum" 
        // https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JA009863
        tem5 = 21.1 * exp(-2.8 * log(Tkininj));
        tem6 = 1.0 + (5.85 * exp(-1.22 * log(Tkininj))) + (1.18 * exp(-2.54 * log(Tkininj)));
        jlis = tem5 / tem6;
        w = jlis / (p * p);
        w = w * p1AU * p1AU;

        // Based on: 
        // Ilya G. Usoskin, Katja Alanko-Huotari, Gennady A. Kovaltsov, Kalevi Mursula. "Heliospheric modulation of cosmic rays: Monthly reconstruction for 1951–2004"
        // https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2005JA011250
        tt = Tkin + Tr;
        t2 = tt + Tr;
        beta = sqrt(Tkin * t2) / tt;
        tem = (Tkin + 0.67) / 1.67;
        tem = exp(-3.93 * log(tem));
        jlisJGR = 2.7e3 * exp(1.12 * log(Tkin)) * tem;
        jlisJGR = jlisJGR / (beta * beta);
        wJGR = jlisJGR / (p * p);
        wJGR = wJGR * p1AU * p1AU;

        // Based on:
        // R. A. Burger, M. S. Potgieter, B. Heber. "Production of cosmogenic Be nuclei in the Earth's atmosphere by cosmic rays: Its dependence on solar modulation and the interstellar cosmic ray spectrum"
        // https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JA009863
        tem = sqrt((Tkin*Tkin) + (2.0*Tkin*0.938));
        jlisBurger = 1.9*1.0e4*exp(-2.78*log(tem));
        tem = sqrt((Tkin*Tkin) + (2.0*Tkin*0.938));
        tem = 0.4866*exp(-2.51*log(tem));
        jlisBurger = jlisBurger/(1 + tem);
        wBurger = jlisBurger/(p*p);
        wBurger = wBurger*p1AU*p1AU;

        for (int ii = 1; ii < 30; ii++)
        {
            if ((Tkininj > SPbins[ii] * 0.99) && (Tkininj < SPbins[ii] * 1.01))
            {
                speSP[ii] = speSP[ii] + w;
                speJGR[ii] = speJGR[ii] + wJGR;
                speBurger[ii] = speBurger[ii] + wBurger;
                speN[ii]++;
            }
        }
        for (int ii = 1; ii < 3; ii++)
        {
            if ((Tkininj > UlyssesBins[ii]) && (Tkininj < UlyssesBins[ii+1]))
            {
                ulyssesBin[ii] = ulyssesBin[ii] + wBurger;
                ulyssesBinN[ii]++;
            }
        }
        for (int ii = 1; ii < 71; ii++)
        {
            if ((Tkininj > AmsBins[ii]) && (Tkininj < AmsBins[ii+1]))
            {
                amsBin[ii] = amsBin[ii] + wBurger;
                amsBinN[ii]++;
            }
        }
    }
    fclose(inputFile);

    struct spectrumOutput spectrumOutput;
    spectrumOutput.fileName = "Usoskin_2005";
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, speN, speJGR, SPECTRUM_SOLARPROP);
    spdlog::info("Spectrum based on Usoskin 2005 has been written to file.");

    spectrumOutput.fileName = "Weber";
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, speN, speSP, SPECTRUM_SOLARPROP);
    spdlog::info("Spectrum based on Weber has been written to file.");

    spectrumOutput.fileName = "Burger";
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, speN, speBurger, SPECTRUM_SOLARPROP);
    spdlog::info("Spectrum based on Burger has been written to file.");

    spectrumOutput.fileName = "Ulysses";
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, ulyssesBinN, ulyssesBin, SPECTRUM_ULYSSES);
    spdlog::info("Spectrum with Ulysses bins based on Burger has been written to file.");
    
    spectrumOutput.fileName = "Ams";
    spectrumOutput.isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(&spectrumOutput, amsBinN, amsBin, SPECTRUM_AMS);
    spdlog::info("Spectrum with AMS bins based on Burger has been written to file.");
}