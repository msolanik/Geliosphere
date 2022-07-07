/**
 * @file TwoDimensionBpAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of 3D B-p method
 * @version 0.2
 * @date 2022-07-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef THREE_DIMENSION_BP_METHOD_H
#define THREE_DIMENSION_BP_METHOD_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 3D B-p simulation.
 * 
 */
class ThreeDimensionBpAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief GPU implementation of 2D B-p method and set input 
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