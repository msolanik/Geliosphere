/**
 * @file BatchRun.hpp
 * @author Tomas Telepcak
 * @brief Abstract definition for algorithm
 * @version 0.1
 * @date 2024-01-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BATCH_RUN
#define BATCH_RUN

#include "AbstractAlgorithm.hpp"


class BatchRun : public AbstractAlgorithm
{
public:
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif