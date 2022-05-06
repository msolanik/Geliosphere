/**
 * @file InteractiveMode.hpp
 * @author Michal Solanik
 * @brief Encapsulates data for interactive mode.
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef INTERACTIVE_MODE_H
#define INTERACTIVE_MODE_H

#include <cuda_runtime.h>
#include "ParamsCarrier.hpp"
#include "CosmicUtils.cuh"

extern "C" void initRNG(curandState_t *state, int blockSize, int threadSize);

/**
 * @brief Interface that is used to define algorithm in context of application.
 * 
 * @details Implementation of this class should define input parameters for simulation, 
 * like memory allocation, choose between CPU and GPU version etc.. 
 * 
 */
class InteractiveMode
{
public:
    InteractiveMode();
    ~InteractiveMode();
    curandState_t * getRNGStateStructure();
private:
    curandState_t *state;

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

#endif // !ABSTRACT_ALGORITHM_H
