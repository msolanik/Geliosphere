/**
 * @file TwoDimensionBpAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of 2D B-p model
 * @version 0.1
 * @date 2022-03-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TWO_DIMENSION_BP_model_H
#define TWO_DIMENSION_BP_model_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 2D B-p simulation.
 * 
 */
class TwoDimensionBpAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief GPU implementation of 2D B-p model and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif