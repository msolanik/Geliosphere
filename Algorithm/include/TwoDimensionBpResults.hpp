/**
 * @file TwoDimensionBpResults.hpp
 * @author Michal Solanik
 * @brief Implementation of 2D B-p method analyzer for output data.
 * @version 0.1
 * @date 2022-03-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TWO_DIMENSION_BP_RESULTS_H
#define TWO_DIMENSION_BP_RESULTS_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface
 * to define analyzer for 2D B-p method for creating output energetic 
 * spectra from log file. 
 * 
 */
class TwoDimensionBpResults : public AbstractAlgorithm
{
public:
	/**
	 * @brief Funtion analyse output log file from 2D B-p simulation and 
	 * create energetic spectra from that. Energetic spectra are 
	 * written into files depending on binning.   
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif
