/**
 * @file OneDimensionFpAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of 1D F-p method
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef FLOAT_FW_H
#define FLOAT_FW_H

#include "AbstractAlgorithm.hpp"
#include "OneDimensionFpSimulation.cuh"
#include "InteractiveMode.hpp"

extern "C" void runFWMethod(simulationInput *simulation);

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 1D F-p simulation.
 * 
 */
class OneDimensionFpAlgorithm : public AbstractAlgorithm
{
public:
	OneDimensionFpAlgorithm(InteractiveMode *interactiveMode = NULL);

	/**
	 * @brief Encapsulates GPU implementation of 1D F-p method and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);

private:

	InteractiveMode *interactiveMode;

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
