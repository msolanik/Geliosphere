/**
 * @file ThreeDimensionBpCpuSimulation.hpp
 * @author Michal Solanik
 * @brief CPU implementation for 3D B-p model
 * @version 0.2
 * @date 2022-07-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef BP_CPU_THREE_DIMENSION_SIMULATION_H
#define BP_CPU_THREE_DIMENSION_SIMULATION_H

#include "AbstractCpuSimulation.hpp"

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
 * to define support functions for running implementation of 2D B-p simulation.
 *
 */
class ThreeDimensionBpCpuSimulation : public AbstractCpuSimulation
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
