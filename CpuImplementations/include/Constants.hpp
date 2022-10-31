/**
 * @file Constants.hpp
 * @author Michal Solanik
 * @brief Constants for CPU implementations.
 * @version 0.2
 * @date 2022-06-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "ParamsCarrier.hpp"

/**
 * @brief Speed of the solar wind in AU/s.
 * 
 */
static double V;

/**
 * @brief Time step of simulation in seconds. 
 * 
 */
static double dt = 5.0;

/**
 * @brief Difusion coeficient in AU^2/s
 * 
 */
static double K0 = 0.000222;

/**
 * @brief Ratio of perpendicular and parallel diffusion.
 * 
 */
static double ratio = 0.2;

/**
 * @brief Constant for polar field correction delta.
 * 
 */
static double delta0 = 8.7e-5;

/**
 * @brief Tilt angle in radians.
 * 
 */
static double alphaM = 5.75*3.1415926535/180.0;

/**
 * @brief Sun polarity.
 * 
 */
const double polarity = 1.0;

/**
 * @brief Proton rest mass.
 * 
 */
const double m0 = 1.67261e-27;

/**
 * @brief The fundamental elementary charge.
 * 
 */
const double q = 1.60219e-19;

/**
 * @brief Speed of light.
 * 
 */
const double c = 2.99793e8;

/**
 * @brief Value of Pi
 * 
 */
const double Pi = 3.1415926535897932384626433832795;

/**
 * @brief Rest energy in [GeV].
 * 
 */
const double T0 = m0 * c * c / (q * 1e9);

/**
 * @brief Rest energy in [J].
 * 
 */
const double T0w = m0 * c * c;

/**
 * @brief Angular velocity of the Sun for rotation period.
 * 
 */
const double omega = 2.866e-6;

/**
 * @brief Scaling of heliospheric magnetic field to value 5 nT at Earth position (1AU, theta=90).
 * 
 */
const double A = 3.4;

/**
 * @brief Units scaling constant.
 * 
 */
const double konvF = 9.0e-5/2.0;

/**
 * @brief Array of bins for 2D models.
 * 
 */
const double SPbins[30] = { 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};

/**
 * @brief Equatoria radius of sun. 
 * 
 */
const double rh =  695510.0/150000000.0; 

/**
 * @brief Set contants for SolarProp-like Backward-in-time model.
 * 
 */
static void setSolarPropConstants(ParamsCarrier *singleTone)
{
	ratio = singleTone->getFloat("solarPropRatio", 0.2f);
}

/**
 * @brief Set contants for Geliosphere Backward-in-time model.
 * 
 */
static void setGeliosphereModelConstants(ParamsCarrier *singleTone)
{
	ratio = singleTone->getFloat("geliosphereRatio", 0.02f);
	delta0 = singleTone->getFloat("C_delta", 8.7e-5f);
	alphaM = singleTone->getFloat("tilt_angle", 0.1f);
	float newK = singleTone->getFloat("K0", singleTone->getFloat("K0_default", -1.0f));
	if (newK != -1.0f && !singleTone->getInt("K0_entered_by_user", 0))
	{
		K0 = singleTone->getFloat("K0_ratio", 5.0f) * newK;
	}
	printf("%g %g %g %g", K0, ratio, delta0, alphaM);
}

/**
 * @brief Set constants values according to data in ParamsCarrier.
 * 
 */
static void setContants(ParamsCarrier *singleTone)
{
	if (singleTone->getString("model", "FWMethod").compare("TwoDimensionBp") == 0)
	{
		setSolarPropConstants(singleTone);
	}
	if (singleTone->getString("model", "FWMethod").compare("ThreeDimensionBp") == 0)
	{
		setGeliosphereModelConstants(singleTone);
	}
	float newDT = singleTone->getFloat("dt", singleTone->getFloat("dt_default", -1.0f));
	if (newDT != -1.0f)
	{
		dt = newDT;
	}
	float newK = singleTone->getFloat("K0", singleTone->getFloat("K0_default", -1.0f));
	if (newK != -1.0f && singleTone->getInt("K0_entered_by_user", 0))
	{
		K0 = newK;
	}
	bool isBackward = (singleTone->getString("model", "FWMethod").compare("BPMethod") == 0);
	float newV = (isBackward) ? singleTone->getFloat("V", singleTone->getFloat("V_default", 1.0f)) * (-1.0f) : singleTone->getFloat("V", singleTone->getFloat("V_default", -1.0f));
	if (newV != -1.0f)
	{
		V = newV;
	}
	else
	{
		V = (isBackward) ? 2.66667e-6 * (-1.0f) : 2.66667e-6;
	}
}