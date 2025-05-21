#ifndef BP_CPU_THREE_DIMENSION_SIMULATION_H
#define BP_CPU_THREE_DIMENSION_SIMULATION_H

#include "AbstractCpuModel.hpp"
#include "IGeliosphereCpuModel.hpp"

#include <mutex>
#include <queue>

struct SimulationOutput {
    double Tkininj;
    double Tkin;
    double r;
    double w;
    double thetainj;
    double theta;
};

class GeliosphereCpuModel : public AbstractCpuModel, public IGeliosphereCpuModel {
public:
    void runSimulation(ParamsCarrier *singleTone);

protected:
    double Beta(double Tkin) override;
    double RigFromTkin(double Tkin) override;
    double W(double p) override;
    double GetMomentum(double Rig) override;
    double LarmorRadius(double Rig, double Bfield) override;
    double AlphaH(double Larmor, double r) override;
    double ComputeF(double theta, double alphaH) override;
    double ComputeFPrime(double theta, double alphaH) override;

private:
    void simulation(int threadNumber, unsigned int availableThreads, int iteration);
    std::mutex outputMutex;
    std::queue<SimulationOutput> outputQueue;
};

#endif
