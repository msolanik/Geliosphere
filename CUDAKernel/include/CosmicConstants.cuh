/**
 * @file CosmicConstants.cuh
 * @author Michal Solanik
 * @brief Constants needed for simulations.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef COSMIC_CONSTANTS_H
#define COSMIC_CONSTANTS_H

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ParamsCarrier.hpp"

/**
 * @brief Speed of the solar wind in AU/s.
 * 
 */
extern __device__ __constant__ float V;

/**
 * @brief Time step in seconds. 
 * 
 */
extern __device__ __constant__ float dt;

/**
 * @brief Difusion coeficient in AU^2/s
 * 
 */
extern __device__ __constant__ float K0;

/**
 * @brief Proton mass.
 * 
 */
extern __device__ __constant__ float m0;

/**
 * @brief The fundamental electrical charge.
 * 
 */
extern __device__ __constant__ float q;

/**
 * @brief Speed of light.
 * 
 */
extern __device__ __constant__ float c;

/**
 * @brief 
 * 
 */
extern __device__ __constant__ float T0;

/**
 * @brief T0 in double precision.
 * 
 */
extern __device__ __constant__ double T0_double;

/**
 * @brief m0 in double precision.
 * 
 */
extern __device__ __constant__ double m0_double;

/**
 * @brief q in double precision.
 * 
 */
extern __device__ __constant__ double q_double;

/**
 * @brief q^2 in double precision.
 * 
 */
extern __device__ __constant__ double q_double_pow;

/**
 * @brief c^2  in double precision.
 * 
 */
extern __device__ __constant__ double c_double;

/**
 * @brief 
 * 
 */
extern __device__ __constant__ float T0w;

/**
 * @brief 
 * 
 */
extern __device__ __constant__ float omega2;

/**
 * @brief Value of Pi
 * 
 */
extern __device__ __constant__ float Pi;

/**
 * @brief Maximal injection energy.
 * 
 */
extern __device__ __constant__ float injectionMax;

/**
 * @brief Amount of particles per energy.
 * 
 */
extern __device__ __constant__ float quantityPerEnergy;

/**
 * @brief Value of injection for theta.
 * 
 */
extern __device__ __constant__ float thetainj;

/**
 * @brief Value of omega.
 * 
 */
extern __device__ __constant__ float omega;

/**
 * @brief Value of ratio.
 * 
 */
extern __device__ __constant__ float ratio;

extern __device__ __constant__ float alphaM;
extern __device__ __constant__ float polarity;
extern __device__ __constant__ float A;
extern __device__ __constant__ float konvF;
extern __device__ __constant__ float driftThetaConstant;
extern __device__ __constant__ float delta0;
extern __device__ __constant__ float rh; 

/**
 * @brief Set constants values according to data in ParamsCarrier.
 * 
 */
void setConstants(ParamsCarrier *singleTone, bool isBackward);

#endif
