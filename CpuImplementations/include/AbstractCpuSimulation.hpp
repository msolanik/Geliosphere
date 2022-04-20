/**
 * @file AbstractCpuMethod.hpp
 * @author Michal Solanik
 * @brief Abstract definition for implementation of model on CPU. 
 * @version 0.2
 * @date 2022-02-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef ABSTRAT_CPU_SIMULATION_H
#define ABSTRAT_CPU_SIMULATION_H

#include "ParamsCarrier.hpp"

/**
 * @brief Interface that is used to define abstract CPU simulation in context of application.
 * 
 * @details Implementation of this class should define simulation and needed operations. 
 * 
 */
class AbstractCpuSimulation
{
public:
	/**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	virtual void runSimulation(ParamsCarrier *singleTone) = 0;
};

#endif // !ABSTRAT_CPU_SIMULATION_H
