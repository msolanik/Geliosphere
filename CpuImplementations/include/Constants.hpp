/**
 * @file Constants.hpp
 * @author Michal Solanik
 * @brief 
 * @version 0.1
 * @date 2022-06-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "ParamsCarrier.hpp"

static double V;
static double dt;
static double K0;
const double m0 = 1.67261e-27;
const double q = 1.60219e-19;
const double c = 2.99793e8;
const double Pi = 3.1415926535897932384626433832795;
const double T0 = m0 * c * c / (q * 1e9);
const double T0w = m0 * c * c;
const double omega = 2.866e-6;
const double ratio = 0.02;
const double alphaM = 5.75*Pi/180.0;            // measured value from experiment
const double polarity = 1.0;                  //  A>0 is 1.0 ; A<0 is -1.0
const double A = 3.4;                        // units  nT AU^2, to have B = 5 nT at Earth (1AU, theta=90)
const double konvF = 9.0e-5/2.0;
const double SPbins[30] = { 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};

static void setContants(ParamsCarrier *singleTone, bool isBackward)
{
	float newDT = singleTone->getFloat("dt", -1.0f);
	if (newDT != -1.0f)
	{
		dt = newDT;
	}
	else
	{
		dt = 5.0;
	}
	float newK = singleTone->getFloat("K0", -1.0f);
	if (newK != -1.0f)
	{
		K0 = newK;
	}
	else
	{
		K0 = 0.000222;
	}
	float newV = (isBackward) ? singleTone->getFloat("V", 1.0f) * (-1.0f) : singleTone->getFloat("V", -1.0f);
	if (newV != -1.0f)
	{
		V = newV;
	}
	else
	{
		V = (isBackward) ? 2.66667e-6 * (-1.0f) : 2.66667e-6;
	}
}