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
    rewind(fin);
    return count;
}

int ResultsUtils::writeSpectrum(struct spectrumOutput *spectrumOutput, double *spectrumCount, double *spectrumValue, enum spectrumType spectrumType)
{
    std::string fileName = spectrumOutput->fileName.append((spectrumOutput->isCsv) ? ".csv" : ".dat");
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
    case SPECTRUM_1E3:
    case SPECTRUM_4E2:
    case SPECTRUM_SOLARPROP:
    case SPECTRUM_AMS:
    case SPECTRUM_ULYSSES:
        fprintf(outputFile, "%s,%s,%s,%s\n", "Tkin", "spe", "speAvg", "speCount");
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
    else if (spectrumType == SPECTRUM_SOLARPROP)
    {
        for (int i = 1; i < 30; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), 0.01f * powf((1.0f + 0.5f), i - 1), spectrumValue[i], 
                (spectrumCount[i] != 0.0) ? spectrumValue[i]/spectrumCount[i] : 0.0, spectrumCount[i]);
        }
    }
    else if (spectrumType == SPECTRUM_ULYSSES) 
    {
        double ulyssesBins[4] = {0.0, 0.125, 0.250, 2.0};
        for (int i = 1; i < 3; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), ulyssesBins[i], spectrumValue[i], 
                (spectrumCount[i] != 0.0) ? spectrumValue[i]/(spectrumCount[i]) : 0.0, spectrumCount[i]);
        }        
    }
    else if (spectrumType == SPECTRUM_AMS) 
    {
        const double amsBins[72] = { 4.924000e-01, 6.207000e-01, 7.637000e-01, 9.252000e-01, 1.105000e+00, 
        1.303000e+00, 1.523000e+00, 1.765000e+00, 2.034000e+00, 2.329000e+00, 2.652000e+00, 3.005000e+00, 
        3.390000e+00, 3.810000e+00, 4.272000e+00, 4.774000e+00, 5.317000e+00, 5.906000e+00, 6.546000e+00, 
        7.236000e+00, 7.981000e+00, 8.787000e+00, 9.653000e+00, 1.060000e+01, 1.160000e+01, 1.264000e+01, 
        1.379000e+01, 1.504000e+01, 1.639000e+01, 1.784000e+01, 1.938000e+01, 2.103000e+01, 2.283000e+01, 
        2.478000e+01, 2.683000e+01, 2.903000e+01, 3.138000e+01, 3.387000e+01, 3.657000e+01, 3.947000e+01, 
        4.257000e+01, 4.587000e+01, 4.942000e+01, 5.322000e+01, 5.727000e+01, 6.162000e+01, 6.632000e+01, 
        7.137000e+01, 7.677000e+01, 8.257000e+01, 8.882000e+01, 9.557000e+01, 1.031000e+02, 1.111000e+02, 
        1.196000e+02, 1.291000e+02, 1.401000e+02, 1.526000e+02, 1.666000e+02, 1.826000e+02, 2.006000e+02, 
        2.211000e+02, 2.451000e+02, 2.741000e+02, 3.096000e+02, 3.536000e+02, 4.091000e+02, 4.821000e+02, 
        5.831000e+02, 7.316000e+02, 9.751000e+02, 1.464000e+03 };
        for (int i = 1; i < 71; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), amsBins[i], spectrumValue[i], 
                (spectrumCount[i] != 0.0) ? spectrumValue[i]/(spectrumCount[i]) : 0.0, spectrumCount[i]);
        }        
    }
    else
    {
        for (int i = 0; i < spectrumOutput->size; i++)
        {
            fprintf(outputFile, getFormat(spectrumType, spectrumOutput->isCsv).c_str(), ((float)(i + 1) / spectrumOutput->tkinPortion), spectrumValue[i], 
                (spectrumCount[i] != 0.0) ? spectrumValue[i]/spectrumCount[i] : 0.0, spectrumCount[i]);
        }
    }
}

std::string ResultsUtils::getFormat(enum spectrumType type, bool isCsv)
{
    switch (type)
    {
    case SPECTRUM_LOG:
        return "%.14E" + getSeparator(isCsv) + "%.14E\n";
    case SPECTRUM_SOLARPROP:
    case SPECTRUM_AMS:
    case SPECTRUM_ULYSSES:
        return "%3.4f" + getSeparator(isCsv) + "%3.4f" + getSeparator(isCsv) + "%3.4f" + getSeparator(isCsv) + "%3.4f\n";
    default:
        return "%3.4f" + getSeparator(isCsv) + "%.14E" + getSeparator(isCsv) + "%.14E\n";
    }
    return NULL;
}

std::string ResultsUtils::getSeparator(bool isCsv)
{
    return (isCsv) ? "," : " ";
}