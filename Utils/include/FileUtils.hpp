/**
 * @file FileUtils.hpp
 * @author Michal Solanik
 * @brief Utilities for manipulating with directories
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>

#include "ParamsCarrier.hpp"

/**
 * @brief Create directory and change active directory
 * on newly created directory
 * 
 * @param directoryName Name of directory to create
 * @return -1 in the case of failure and 
 * @return 1 in the case of success.  
 */
int mkdirAndchdir(std::string directoryName);

/**
 * @brief Create a Directory object
 * 
 * @param methodDirectory Name of directory for used method.
 * @param destination Name of new directory
 * @return true in the case of success
 * @return false in the case of failure and 
 */
bool createDirectory(std::string methodDirectory, std::string destination);

/**
 * @brief Get the Directory Name object
 * 
 * @param singleTone data structure containing all parameters.
 * @return Return directory name with parameters.
 */
std::string getDirectoryName(ParamsCarrier *singleTone);

/**
 * @brief Write simulation report to file
 * 
 * @param singleTone data structure containing all parameters.
 * 
 */
void writeSimulationReportFile(ParamsCarrier *singleTone);

#endif