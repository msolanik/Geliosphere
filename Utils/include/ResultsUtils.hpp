/**
 * @file ResultsUtils.hpp
 * @author Michal Solanik
 * @brief Utility for analyting log files. 
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef RESULTS_UTILS_H
#define RESULTS_UTILS_H

#include <string>
#include <cstdio>
#include <cstdbool>

#define BUFF_SIZE 1024

/**
 * @brief Enum representing type of spectrum.
 * 
 */
enum spectrumType
{
    SPECTRUM_LOG = 1,
    SPECTRUM_1E2 = 2,
    SPECTRUM_1E3 = 3,
    SPECTRUM_4E2 = 4
};

/**
 * @brief Data structure representing data needed for generating energetic 
 * spectra output.
 * 
 */
struct spectrumOutput
{
    std::string fileName;
    int size;
    int tkinPortion;
    bool isCsv;
};

/**
 * @brief This class is used for binning data and creating output 
 * energetic spectra.
 * 
 */
class ResultsUtils
{
    public:
        /**
         * @brief Print header for .csv file.
         * 
         * @param outputFile File where header should be written
         * @param spectrumType Type of energetic spectra
         */
        void printCsvHeader(FILE *outputFile, enum spectrumType spectrumType);

        /**
         * @brief Write spectrum to file. 
         * 
         * @param spectrumOutput data structure containg data needed for creating output
         * spectra.
         * @param spectrumCount Array of counted energies on individuals bins.
         * @param spectrumValue Array of kinetic energies on individual bins.
         * @param spectrumType Type of energetic spectra.
         * @return 0 in the case of success write to file.
         */
        int writeSpectrum(struct spectrumOutput *spectrumOutput, double *spectrumCount, double *spectrumValue, enum spectrumType spectrumType);

        /**
         * @brief Count lines in given file.
         * 
         * @param fin File where lines should be counted.
         * @return number of lines in file.
         */
        int countLines(FILE *const fin);
    private: 
        /**
         * @brief Write spectrum to file.
         * 
         * @param spectrumOutput data structure containg data needed for creating output
         * spectra.
         * @param outputFile Output file where energetic spectra will be written.
         * @param spectrumCount Array of counted energies on individuals bins.
         * @param spectrumValue Array of kinetic energies on individual bins.
         * @param spectrumType Type of energetic spectra.
         */
        void writeSpectrumToFile(struct spectrumOutput *spectrumOutput, FILE *outputFile, double *spectrumCount, double *spectrumValue, enum spectrumType spectrumType);

        /**
         * @brief Get the format for writing to file.
         * 
         * @param spectrumType Type of energetic spectra. 
         * @param isCsv Flag representing if format should be in .csv format.
         * @return Format for given energetic spectra type.
         */
        std::string getFormat(enum spectrumType spectrumType, bool isCsv);

        /**
         * @brief Get the separator based on .dat or .csv format
         * 
         * @param isCsv Flag representing if separator should be in .csv format.
         * @return Separator for given type of file. 
         */
        std::string getSeparator(bool isCsv);
};

#endif