/**
 * @file OneDimensionBpAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of 1D B-p method
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef BP_METHOD_H
#define BP_METHOD_H

#include "AbstractAlgorithm.hpp"
#include "OneDimensionBpSimulation.cuh"

extern "C" void runBPMethod(simulationInputBP *simulation);

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 1D B-p simulation.
 * 
 */
class OneDimensionBpAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief GPU implementation of 1D B-p method and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);

private:
	/**
	 * @brief Define number of threads in block.  
	 * 
	 */
	int threadSize;
	
	/**
	 * @brief Define number of blocks in grid.  
	 * 
	 */
	int blockSize;
	
	/**
	 * @brief Set size for @ref blockSize and @ref threadSize.
	 * 
	 */
	void setThreadBlockSize();
};

#endif