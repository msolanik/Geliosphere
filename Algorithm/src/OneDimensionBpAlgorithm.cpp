#include "OneDimensionBpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "OneDimensionBpResults.hpp"
#include "OneDimensionBpCpuSimulation.hpp"
#if GPU_ENABLED == 1
#include "OneDimensionBpGpuSimulation.hpp"
#endif

void OneDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
    	OneDimensionBpGpuSimulation *oneDimensionBpGpuSimulation = new OneDimensionBpGpuSimulation();
		oneDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif		
	}
	else
	{
		OneDimensionBpCpuSimulation *oneDimensionBpCpuSimulation = new OneDimensionBpCpuSimulation();
		oneDimensionBpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new OneDimensionBpResults();
	result->runAlgorithm(singleTone);
}