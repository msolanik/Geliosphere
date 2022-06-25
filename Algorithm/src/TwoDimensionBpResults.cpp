#include "TwoDimensionBpResults.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "ResultsUtils.hpp"

#define bufSize 1024
#define TARGET 100000

const static double m0 = 1.67261e-27;
const static double q = 1.60219e-19;
const static double c = 2.99793e8;
const static double T0w = m0 * c * c;
const static double Tr = 0.938;
const double SPbins[31] = { 0, 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};

void TwoDimensionBpResults::runAlgorithm(ParamsCarrier *singleTone)
{
    ResultsUtils *resultsUtils = new ResultsUtils();
    int energy1e2, energy1e3, energy4e2;
    double w, Rig, p1AU, Tkin, r, p, Tkinw, Rig1AU, Tkininj, theta, thetainj, tt, t2, beta;
    double tem6, tem5, jlis, tem, wJGR;
    double binb[301], binw[301], binc[301];
    double spe1e2[101] = {0}, spe1e2N[101] = {0};
    double spe4e2[401] = {0}, spe4e2N[401] = {0};
    double spe1e3[1001] = {0}, spe1e3N[1001] = {0};
    double spelog[301] = {0}, spelogN[301] = {0};
    double speSP[31] = {0}, speJGR[31] = {0};
    int NT = 80;
    double Tmin = 0.01;
    double Tmax = 200.0;
    double dlT = log10(Tmax / Tmin) / NT;
    double X = log10(Tmax);
    double larg = exp((dlT / 2.0) * log(10.0)) - exp((-1.0 * dlT / 2.0) * log(10.0));
    for (int i = 0; i < NT; i++)
    {
        spelog[i] = 0;
        double tem = X - (i * dlT);
        binb[i] = exp(tem * log(10.0));
        binc[i] = sqrt(binb[i] * exp((tem + dlT) * log(10.0)));
        binw[i] = larg * binc[i];
    }
    FILE *inputFile = fopen("log.dat", "r");
    int numberOfIterations = resultsUtils->countLines(inputFile) - 1;
    int targetArray[] = {numberOfIterations};
    FILE *outputFile;
    for (int j = 0; j < 1; ++j)
    {
        rewind(inputFile);
        int targetLine = targetArray[j];
        int actualLine = 0;
        for (int i = 0; i < numberOfIterations / targetArray[j]; i++)
        {
            actualLine = 0;
            while (targetLine != actualLine)
            {
                ++actualLine;
                int reader = fscanf(inputFile, "%lf %lf %lf %lf %lf %lf\n", &Tkininj, &Tkin, &r, &w, &thetainj, &theta);
                if (reader == -1)
                {
                    return;
                }
                Tkinw = Tkin * 1e9 * q;
                Rig = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
                p = Rig * q / c;
                Tkinw = Tkininj * 1e9 * q;
                Rig1AU = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
                p1AU = Rig1AU * q / c;
                

                tem5 = 21.1*exp(-2.8*log(Tkininj));
                tem6 = 1 + (5.85*exp(-1.22*log(Tkininj))) + (1.18*exp(-2.54*log(Tkininj)));
                jlis = tem5/tem6;                            
                w = jlis/(p*p);                        
                w = w*p1AU*p1AU;

                // Ako to je s JGR? 
                tt = Tkin + Tr;
                t2 = tt + Tr;
                beta = sqrt(Tkin * t2) / tt;
                tem = (Tkin + 0.67)/1.67;
                tem = exp(-3.93*log(tem));
                jlis = 2.7e3 * exp(1.12*log(Tkin)) * tem ;
                jlis = jlis / (beta *beta);
            
                wJGR = jlis/(p*p);
                wJGR = wJGR*p1AU*p1AU;
                for (int ii = 0; ii < 299; ii++)
                {
                    if ((Tkin > binb[ii + 1]) && (Tkin < binb[ii]))
                    {
                        spelog[ii + 1] = spelog[ii + 1] + w;
                        spelogN[ii + 1] = spelogN[ii + 1] + 1;
                    }
                }
                for (int ii = 0; ii < 30; ii++)
                {
                    if ((Tkininj>SPbins[ii]*0.9) && (Tkininj<SPbins[ii]*1.1))
                    {
                        speSP[ii + 1] = speSP[ii + 1] + w;
                        speJGR[ii + 1] = speJGR[ii + 1] + wJGR;
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
        }
    }
    fclose(inputFile);
    FILE *out = fopen("JGAR.csv", "w");; 
    for (int i = 1; i < 31; i++)
    {
        fprintf(out, "%3.4f,%3.4f\n", SPbins[i], speJGR[i]);
    }
    fclose(out);

    out = fopen("Weber.csv", "w");; 
    for (int i = 1; i < 31; i++)
    {
        fprintf(out, "%3.4f,%3.4f\n", SPbins[i], speSP[i]);
    }
    fclose(out);

    // struct spectrumOutput *spectrumOutput;
    // spectrumOutput->fileName = "output_1e3bin";
    // spectrumOutput->size = 1000;
    // spectrumOutput->tkinPortion = 10;
    // spectrumOutput->isCsv = singleTone->getInt("csv", 0);
    // resultsUtils->writeSpectrum(spectrumOutput, spe1e3N, spe1e3, SPECTRUM_1E3);

    // spectrumOutput->fileName = "output_1e2bin";
    // spectrumOutput->size = 100;
    // spectrumOutput->tkinPortion = 1;
    // resultsUtils->writeSpectrum(spectrumOutput, spe1e2N, spe1e2, SPECTRUM_1E2);

    // spectrumOutput->fileName = "output_logbin";
    // resultsUtils->writeSpectrum(spectrumOutput, NULL, spelog, SPECTRUM_LOG);

    // spectrumOutput->fileName = "output_4e2bin";
    // spectrumOutput->size = 400;
    // spectrumOutput->tkinPortion = 4;
    // resultsUtils->writeSpectrum(spectrumOutput, spe4e2N, spe4e2, SPECTRUM_4E2);
}