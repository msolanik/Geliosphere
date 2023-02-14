/**
 * @file ThreeDimensionBpGpuSimulation.hpp
 * @author Michal Solanik
 * @brief GPU implementation for 3D B-p model
 * @version 0.2
 * @date 2022-08-15
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef BP_GPU_THREE_DIMENSION_SIMULATION_H
#define BP_GPU_THREE_DIMENSION_SIMULATION_H

#include "AbstractGpuSimulation.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 3D B-p simulation.
 * 
 */
class ThreeDimensionBpGpuSimulation : public AbstractGpuSimulation
{
public:
    /**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	void prepareAndRunSimulation(ParamsCarrier *singleTone);
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
	 * @brief Define maximum size of a shared memory.  
	 * 
	 */
	int sharedMemoryMaximumSize;
	
	/**
	 * @brief Set size for @ref blockSize and @ref threadSize.
	 * 
	 */
	void setThreadBlockSize();
};

#endif
