/**
 * @file OneDimensionFpCpuModel.hpp
 * @author Michal Solanik
 * @brief CPU implementation for 1D F-p model
 * @version 0.2
 * @date 2022-06-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef BP_CPU_ONE_DIMENSION_SIMULATION_H
#define BP_CPU_ONE_DIMENSION_SIMULATION_H

#include "AbstractCpuModel.hpp"

#include <mutex>
#include <queue>

struct SimulationOutput {
	double Tkin;
	double Tkininj;
	double r;
	double w;
};

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of 1D B-p model.
 * 
 */
class OneDimensionBpCpuModel : public AbstractCpuModel
{
public:
    /**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	void runSimulation(ParamsCarrier *singleTone);
private:
    void simulation(int threadNumber, unsigned int availableThreads, int iteration);
    std::mutex outputMutex;
    std::queue<SimulationOutput> outputQueue;
};

#endif
