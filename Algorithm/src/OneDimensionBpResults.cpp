#include "OneDimensionBpResults.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "ResultsUtils.hpp"

#define bufSize 1024
#define TARGET 100000

static double m0 = 1.67261e-27;
static double q = 1.60219e-19;
static double c = 2.99793e8;
static double T0w = m0 * c * c;

void OneDimensionBpResults::runAlgorithm(ParamsCarrier *singleTone)
{
    ResultsUtils *resultsUtils = new ResultsUtils();
    int energy1e2, energy1e3, energy4e2;
    double w, Rig, p1AU, Tkin, r, p, Tkinw, Rig1AU, Tkininj;
    double binb[300], binw[300], binc[300];
    double spe1e2[100] = {0}, spe1e2N[100] = {0};
    double spe4e2[400] = {0}, spe4e2N[400] = {0};
    double spe1e3[1000] = {0}, spe1e3N[1000] = {0};
    double spelog[300] = {0}, spelogN[300] = {0};
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
                int reader = fscanf(inputFile, "%lf %lf %lf %lf\n", &Tkininj, &Tkin, &r, &w);
                if (reader == -1)
                {
                    return;
                }
                Tkinw = Tkininj * 1e9 * q;
                Rig = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
                p = Rig * q / c;
                Tkinw = Tkin * 1e9 * q;
                Rig1AU = sqrt(Tkinw * (Tkinw + (2 * T0w))) / q;
                p1AU = Rig1AU * q / c;
                w = (m0 * m0 * c * c * c * c) + (p * p * c * c);
                w = exp(-1.85 * log(w)) / p;
                w = w * p1AU * p1AU;
                for (int ii = 0; ii < 299; ii++)
                {
                    if ((Tkin > binb[ii + 1]) && (Tkin < binb[ii]))
                    {
                        spelog[ii + 1] = spelog[ii + 1] + w;
                        spelogN[ii + 1] = spelogN[ii + 1] + 1;
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

    struct spectrumOutput *spectrumOutput;
    spectrumOutput->fileName = "output_1e3bin";
    spectrumOutput->size = 1000;
    spectrumOutput->tkinPortion = 10;
    spectrumOutput->isCsv = singleTone->getInt("csv", 0);
    resultsUtils->writeSpectrum(spectrumOutput, spe1e3N, spe1e3, SPECTRUM_1E3);

    spectrumOutput->fileName = "output_1e2bin";
    spectrumOutput->size = 100;
    spectrumOutput->tkinPortion = 1;
    resultsUtils->writeSpectrum(spectrumOutput, spe1e2N, spe1e2, SPECTRUM_1E2);

    spectrumOutput->fileName = "output_logbin";
    resultsUtils->writeSpectrum(spectrumOutput, NULL, spelog, SPECTRUM_LOG);

    spectrumOutput->fileName = "output_4e2bin";
    spectrumOutput->size = 400;
    spectrumOutput->tkinPortion = 4;
    resultsUtils->writeSpectrum(spectrumOutput, spe4e2N, spe4e2, SPECTRUM_4E2);
}