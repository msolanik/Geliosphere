/**
 * @file CudaErrorCheck.cuh
 * @author Michal Solanik
 * @brief Utility for checking errors.
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

/**
 * @brief Macro for checking errors on GPU.
 * 
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief Log error on stderr and exit application after failure.
 * 
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU fail: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
		{
			exit(code);
		}
	}
}

#endif // !CUDA_ERROR_CHECK_H
