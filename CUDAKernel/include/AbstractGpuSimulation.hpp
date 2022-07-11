/**
 * @file AbstractGpuMethod.hpp
 * @author Michal Solanik
 * @brief Abstract definition for implementation of model on GPU. 
 * @version 0.2
 * @date 2022-07-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef ABSTRACT_GPU_SIMULATION_H
#define ABSTRACT_GPU_SIMULATION_H

#include "ParamsCarrier.hpp"

/**
 * @brief Interface that is used to define abstract GPU simulation in context of application.
 * 
 * @details Implementation of this class should define simulation and needed operations. 
 * 
 */
class AbstractGpuSimulation
{
public:
	/**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	virtual void prepareAndRunSimulation(ParamsCarrier *singleTone) = 0;
};

#endif // !ABSTRACT_CPU_SIMULATION_H
