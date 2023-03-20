/**
 * @file SolarPropLikeCpuModel.hpp
 * @author Michal Solanik
 * @brief CPU implementation for 2D B-p model
 * @version 0.2
 * @date 2022-02-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef BP_CPU_TWO_DIMENSION_SIMULATION_H
#define BP_CPU_TWO_DIMENSION_SIMULATION_H

#include "AbstractCpuModel.hpp"

#include <mutex>
#include <queue>

struct SimulationOutput
{
    double Tkininj;
    double Tkin;
    double r;
    double w;
    double thetainj;
    double theta;
};

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface
 * to define support functions for running implementation of SolarProp-like 2D B-p model.
 *
 */
class SolarPropLikeCpuModel : public AbstractCpuModel
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
