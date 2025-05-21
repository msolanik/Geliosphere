#ifndef I_ONE_DIMENSION_FP_CPU_MODEL_H
#define I_ONE_DIMENSION_FP_CPU_MODEL_H

class IOneDimensionFpCpuModel {
protected:
    virtual double Beta(double Tkin) = 0;
    virtual double RigFromTkin(double Tkin) = 0;
    virtual double RigFromTkinJoule(double Tkin) = 0;
    virtual double Kdiffr(double beta, double Rig) = 0;
    virtual double Dp(double V, double p, double r) = 0;
    virtual double Dr(double V, double Kdiff, double r, double dt, double rand) = 0;
    virtual double Cfactor(double V, double r) = 0;
    virtual double TkinFromRig(double Rig) = 0;
    virtual double W(double p) = 0;

public:
    virtual ~IOneDimensionFpCpuModel() = default;
};

#endif // I_ONE_DIMENSION_FP_CPU_MODEL_H
