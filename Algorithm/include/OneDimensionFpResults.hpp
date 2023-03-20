/**
 * @file OneDimensionFpResults.hpp
 * @author Michal Solanik
 * @brief Implementation of 1D F-p model analyzer for output data.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef FLOAT_FW_RESULT_H
#define FLOAT_FW_RESULT_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface
 * to define analyzer for 1D F-p model for creating output energetic 
 * spectra from log file. 
 * 
 */
class OneDimensionFpResults : public AbstractAlgorithm
{
public:
	/**
	 * @brief Funtion analyse output log file from 1D F-p simulation and 
	 * create energetic spectra from that. Energetic spectra are 
	 * written into files depending on binning.   
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif
