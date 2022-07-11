#include "TwoDimensionBpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "TwoDimensionBpResults.hpp"
#include "TwoDimensionBpCpuSimulation.hpp"
#if GPU_ENABLED == 1
#include "TwoDimensionBpGpuSimulation.hpp"
#endif

void TwoDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
		TwoDimensionBpGpuSimulation *twoDimensionBpGpuSimulation = new TwoDimensionBpGpuSimulation();
		twoDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif		
	}
	else
	{
		TwoDimensionBpCpuSimulation *twoDimensionBpCpuSimulation = new TwoDimensionBpCpuSimulation();
		twoDimensionBpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);
}