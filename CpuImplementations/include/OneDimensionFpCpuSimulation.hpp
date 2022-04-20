/**
 * @file OneDimensionFpCpuSimulation.hpp
 * @author Michal Solanik
 * @brief CPU implementation for 1D F-p model
 * @version 0.2
 * @date 2022-02-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef FP_CPU_ONE_DIMENSION_SIMULATION_H
#define FP_CPU_ONE_DIMENSION_SIMULATION_H

#include "AbstractCpuSimulation.hpp"

#include <mutex>
#include <queue>

struct SimulationOutput {
    double pinj;
    double p;
    double r;
    double w;
    double sumac;
};

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 1D F-p simulation.
 * 
 */
class OneDimensionFpCpuSimulation : public AbstractCpuSimulation
{
public:
    /**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	void runSimulation(ParamsCarrier *singleTone);
private:
    void simulation();
    std::mutex outputMutex;
    std::queue<SimulationOutput> outputQueue;
};

#endif
