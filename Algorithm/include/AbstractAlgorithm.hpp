/**
 * @file AbstractAlgorithm.hpp
 * @author Michal Solanik
 * @brief Abstract definition for algorithm
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef ABSTRACT_ALGORITHM_H
#define ABSTRACT_ALGORITHM_H

#include "ParamsCarrier.hpp"

/**
 * @brief Interface that is used to define algorithm in context of application.
 * 
 * @details Implementation of this class should define input parameters for simulation, 
 * like memory allocation, choose between CPU and GPU version etc.. 
 * 
 */
class AbstractAlgorithm
{
public:
	/**
	 * @brief Definition of method that run implemented algorithm.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	virtual void runAlgorithm(ParamsCarrier *singleTone) = 0;
};

#endif // !ABSTRACT_ALGORITHM_H
