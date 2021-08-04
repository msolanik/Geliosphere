/**
 * @file CosmicUtils.cuh
 * @author Michal Solanik
 * @brief Common functions for simulations.
 * @version 0.1
 * @date 2021-07-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef COSMIC_UTILS_H
#define COSMIC_UTILS_H

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

/**
 * @brief Counter pointing at current position in output structure.
 * 
 */
extern __device__ int outputCounter;

/**
 * @brief Get value of injection kinetic energy
 * 
 * @param state State of order that is used to calculate kinetic energy
 * @return Value of kinetic energy
 */
__device__ float getTkinInjection(unsigned long long state);

/**
 * @brief Set counter to 0
 * 
 */
__global__ void nullCount();

/**
 * @brief Initialize random number generators.
 * 
 * @param state Array of random number generator data structures.
 */
__global__ void curandInitialization(curandState_t *state);

#endif 