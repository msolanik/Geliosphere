/**
 * @file OneDimensionFpCpuModel.hpp
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

#include "AbstractCpuModel.hpp"
#include "IOneDimensionFpCpuModel.hpp"

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
 * to define support functions for running implementation of 1D F-p model.
 * 
 */
class OneDimensionFpCpuModel : public AbstractCpuModel, public IOneDimensionFpCpuModel
{
public:
    /**
	 * @brief Definition of simulation runner.
	 * 
	 * @param singleTone data structure containing input parameters.
	 */
	void runSimulation(ParamsCarrier *singleTone);

protected:
	// Implemented from interface
    double Beta(double Tkin) override;
    double RigFromTkin(double Tkin) override;
    double RigFromTkinJoule(double Tkin) override;
    double Kdiffr(double beta, double Rig) override;
    double Dp(double V, double p, double r) override;
    double Dr(double V, double Kdiff, double r, double dt, double rand) override;
    double Cfactor(double V, double r) override;
    double TkinFromRig(double Rig) override;
    double W(double p) override;

private:
    void simulation(int threadNumber, unsigned int availableThreads, int iteration);
    std::mutex outputMutex;
    std::queue<SimulationOutput> outputQueue;
};

#endif
