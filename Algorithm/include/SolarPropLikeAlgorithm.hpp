/**
 * @file SolarPropLikeAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of SOLARPROPLike model algorithm
 * @version 0.1
 * @date 2022-03-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef SOLARPROPLIKE_MODEL_H
#define SOLARPROPLIKE_MODEL_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of SOLARPROPLike model simulation.
 * 
 */
class SolarPropLikeAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief GPU implementation of SOLARPROPLike model and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif