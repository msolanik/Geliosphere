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
#include "IOneDimensionBpCpuModel.hpp"

#include <mutex>
#include <queue>

struct SimulationOutput {
	double Tkin;
	double Tkininj;
	double r;
	double w;
};


class OneDimensionBpCpuModel : public AbstractCpuModel, public IOneDimensionBpCpuModel
{
public:

	void runSimulation(ParamsCarrier *singleTone);

protected:    
    // Implemented from interface
    double Beta(double Tkin) override;
    double RigFromTkin(double Tkin) override;
    //double RigFromMomentum(double p) override;
    double Kdiffr(double beta, double Rig) override;
    double Dp(double V, double p, double r) override;
    double Dr(double V, double Kdiff, double r, double dt, double rand) override;
    //double TkinFromRig(double Rig) override;
    double W(double p) override;

private:
    void simulation(int threadNumber, unsigned int availableThreads, int iteration);
    std::mutex outputMutex;
    std::queue<SimulationOutput> outputQueue;
};

#endif
 
