/**
 * @file CosmicConstants.cu
 * @author Michal Solanik
 * @brief File contains implementation of manipulation with constants.
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "CosmicConstants.cuh"
#include "ParamsCarrier.hpp"

/**
 * @brief Set constants values according to data in ParamsCarrier.
 * 
 */
extern "C" void setConstants(ParamsCarrier *singleTone, bool isBackward);

__device__ __constant__ float V = 2.66667e-6;
__device__ __constant__ float dt = 5.0f;
__device__ __constant__ float K0 = 0.000222;
__device__ __constant__ float m0 = 1.67261e-27;
__device__ __constant__ float q = 1.60219e-19;
__device__ __constant__ double m0_double = 1.67261e-27;
__device__ __constant__ double q_double = 1.60219e-19;
__device__ __constant__ double q_double_pow = 1.60219e-19 * 1.60219e-19;
__device__ __constant__ double c_double = 2.99793e8;
__device__ __constant__ float c = 2.99793e8;
__device__ __constant__ float T0 = 1.67261e-27 * 2.99793e8 * 2.99793e8 / (1.60219e-19 * 1e9);
__device__ __constant__ double T0_double = 1.67261e-27 * 2.99793e8 * 2.99793e8 / (1.60219e-19 * 1e9);
__device__ __constant__ float T0w = 1.67261e-27 * 2.99793e8 * 2.99793e8;
__device__ __constant__ float omega2 = 7.25445387e-12;
__device__ __constant__ float Pi = 3.141592654;
__device__ __constant__ float injectionMax = 150.0f;
__device__ __constant__ float quantityPerEnergy = 10000.0f;
__device__ __constant__ float thetainj = 3.1415926535f / 2.0f;
__device__ __constant__ float omega = 2.866e-6f;
__device__ __constant__ float ratio = 0.02f;
__device__ __constant__ float alphaM = 5.75f * 3.1415926535f / 180.0f;
__device__ __constant__ float polarity = 1.0f;
__device__ __constant__ float A = 3.4f;
__device__ __constant__ float konvF = 9.0e-5f/2.0f;
__device__ __constant__ float driftThetaConstant = -1.0f*1.0f*(9.0e-5f/2.0f)*(2.0f/(3.0f*3.4f));

void setConstants(ParamsCarrier *singleTone, bool isBackward)
{
	float newDT = singleTone->getFloat("dt", -1.0f);
	if (newDT != -1.0f)
	{
		cudaMemcpyToSymbol(dt, &newDT, sizeof(newDT));
	}
	float newK = singleTone->getFloat("K0", -1.0f);
	if (newK != -1.0f)
	{
		cudaMemcpyToSymbol(K0, &newK, sizeof(newK));
	}
	float newV = (isBackward) ? singleTone->getFloat("V", 1.0f) * (-1.0f) : singleTone->getFloat("V", -1.0f);
	if (newV != -1.0f)
	{
		cudaMemcpyToSymbol(V, &newV, sizeof(newV));
	}
	else
	{
		newV = (isBackward) ? 2.66667e-6 * (-1.0f) : 2.66667e-6;
		cudaMemcpyToSymbol(V, &newV, sizeof(newV));
	}
}