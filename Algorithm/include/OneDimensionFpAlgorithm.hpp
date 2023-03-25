/**
 * @file OneDimensionFpAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of 1D F-p model
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef FLOAT_FW_H
#define FLOAT_FW_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 1D F-p simulation.
 * 
 */
class OneDimensionFpAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief Encapsulates GPU implementation of 1D F-p model and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif
