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
 * @brief Time step of simulation in seconds. 
 * 
 */
extern __device__ __constant__ float dt;

/**
 * @brief Difusion coeficient in AU^2/s
 * 
 */
extern __device__ __constant__ float K0;

/**
 * @brief Proton rest mass.
 * 
 */
extern __device__ __constant__ float m0;

/**
 * @brief The fundamental elementary charge.
 * 
 */
extern __device__ __constant__ float q;

/**
 * @brief Speed of light.
 * 
 */
extern __device__ __constant__ float c;

/**
 * @brief Rest energy in [GeV].
 * 
 */
extern __device__ __constant__ float T0;

/**
 * @brief Rest energy in double precision.
 * 
 */
extern __device__ __constant__ double T0_double;

/**
 * @brief Proton rest mass in double precision.
 * 
 */
extern __device__ __constant__ double m0_double;

/**
 * @brief The fundamental elementary charge in double precision.
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
 * @brief Rest energy in [J].
 * 
 */
extern __device__ __constant__ float T0w;

/**
 * @brief Value of Pi.
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
 * @brief Angular velocity of the Sun for rotation period.
 * 
 */
extern __device__ __constant__ float omega;

/**
 * @brief Ratio of perpendicular and parallel diffusion.
 * 
 */
extern __device__ __constant__ float ratio;

/**
 * @brief Tilt angle in radians.
 * 
 */
extern __device__ __constant__ float alphaM;

/**
 * @brief Sun polarity.
 * 
 */
extern __device__ __constant__ float polarity;

/**
 * @brief Scaling of heliospheric magnetic field to value 5 nT at Earth position (1AU, theta=90).
 * 
 */
extern __device__ __constant__ float A;

/**
 * @brief Units scaling constant.
 * 
 */
extern __device__ __constant__ float konvF;

/**
 * @brief Constant part of calculating drift for theta. 
 * 
 */
extern __device__ __constant__ float driftThetaConstant;

/**
 * @brief Constant for polar field correction delta.
 * 
 */
extern __device__ __constant__ float delta0;

/**
 * @brief Equatoria radius of sun. 
 * 
 */
extern __device__ __constant__ float rh; 

/**
 * @brief Initial value of r.
 * 
 */
extern __device__ __constant__ float rInit;

/**
 * @brief Set constants values according to data in ParamsCarrier.
 * 
 */
void setConstants(ParamsCarrier *singleTone);

/**
 * @brief Set contants for SolarProp-like Backward-in-time model.
 * 
 */
void setSolarPropConstants(ParamsCarrier *singleTone);

/**
 * @brief Set contants for Geliosphere Backward-in-time model.
 * 
 */
void setGeliosphereModelConstants(ParamsCarrier *singleTone);

#endif
