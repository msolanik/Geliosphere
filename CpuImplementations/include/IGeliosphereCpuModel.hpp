#ifndef I_GELIOSPHERE_CPU_MODEL_H
#define I_GELIOSPHERE_CPU_MODEL_H

class IGeliosphereCpuModel {
protected:
    virtual double Beta(double Tkin) = 0;
    virtual double RigFromTkin(double Tkin) = 0;
    virtual double W(double p) = 0;
    virtual double GetMomentum(double Rig) = 0;
    virtual double LarmorRadius(double Rig, double Bfield) = 0;
    virtual double AlphaH(double Larmor, double r) = 0;
    virtual double ComputeF(double theta, double alphaH) = 0;
    virtual double ComputeFPrime(double theta, double alphaH) = 0;

public:
    virtual ~IGeliosphereCpuModel() = default;
};

#endif // I_GELIOSPHERE_CPU_MODEL_H
