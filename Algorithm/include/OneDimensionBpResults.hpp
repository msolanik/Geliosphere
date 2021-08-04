/**
 * @file OneDimensionBpResults.hpp
 * @author Michal Solanik
 * @brief Implementation of 1D B-p method analyzer for output data.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef BP_RES_H
#define BP_RES_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface
 * to define analyzer for 1D B-p method for creating output energetic 
 * spectra from log file. 
 * 
 */
class OneDimensionBpResults : public AbstractAlgorithm
{
public:
	/**
	 * @brief Funtion analyse output log file from 1D B-p simulation and 
	 * create energetic spectra from that. Energetic spectra are 
	 * written into files depending on binning.   
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif
