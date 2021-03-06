#include <math.h>
#include "ResultsUtils.hpp"

int ResultsUtils::countLines(FILE *const fin)
{
    char buf[BUFF_SIZE];
    int count = 0;
    while (fgets(buf, sizeof(buf), fin) != NULL)
    {
        ++count;
    }
    return count;
}

int ResultsUtils::writeSpectrum(struct spectrumOutput *spectrumOutput, double *spectrumCount, double *spectrumValue, enum spectrumType spectrumType)
{
    std::string fileName = spectrumOutput->fileName + (spectrumOutput->isCsv) ? ".csv" : ".dat";
    FILE *outputFile = fopen(fileName.c_str(), "w");
    if (spectrumOutput->isCsv)
    {
        printCsvHeader(outputFile, spectrumType);
    }
    writeSpectrumToFile(spectrumOutput, outputFile, spectrumCount, spectrumValue, spectrumType);
    fclose(outputFile);
    return 0;
}

void ResultsUtils::printCsvHeader(FILE *outputFile, enum spectrumType spectrumType)
{
    switch (spectrumType)
    {
    case SPECTRUM_LOG:
        fprintf(outputFile, "%s,%s\n", "binc", "spelog/binw");
        break;
    case SPECTRUM_1E2:
        fprintf(outputFile, "%s,%s,%s\n", "Tkin", "spe1e2N", "spe1e2");
        break;
    case SPECTRUM_1E3:
        fprintf(outputFile, "%s,%s,%s\n", "Tkin", "spe1e3N", "spe1e3");
        break;
    case SPECTRUM_4E2:
        fprintf(outputFile, "%s,%s,%s\n", "Tkin", "spe4e2N", "spe4e2");
        break;
    }
}

void ResultsUtils::writeSpectrumToFile(struct spectrumOutput *spectrumOutput, FILE *outputFile, double *spectrumCount, double *spectrumValue, enum spectrumType spectrumType)
{
    if (spectrumType == SPECTRUM_LOG)
    {
        double binb[25], binw[25], binc[25];
        for (int i = 0; i < 25; i++)
        {
            binb[i] = exp((((i + 1) / 5.0) - 3.0) * log(10));
        }
        for (int i = 0; i < 24; i++)
        {
            binw[i] = binb[i + 1] - binb[i];
            binc[i] = (binb[i + 1] + binb[i]) / 2.0;
        }
        for (int i = 0; i < 24; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), binc[i], spectrumValue[i] / binw[i]);
        }
    }
    else
    {
        for (int i = 0; i < spectrumOutput->size; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), ((float)(i + 1) / spectrumOutput->tkinPortion), spectrumCount[i], spectrumValue[i]);
        }
    }
}

std::string ResultsUtils::getFormat(enum spectrumType type, bool isCsv)
{
    switch (type)
    {
    case SPECTRUM_LOG:
        return "%.14E" + getSeparator(isCsv) + "%.14E\n";
    default:
        return "%3.4f" + getSeparator(isCsv) + "%.14E" + getSeparator(isCsv) + "%.14E\n";
    }
    return NULL;
}

std::string ResultsUtils::getSeparator(bool isCsv)
{
    return (isCsv) ? "," : " ";
}