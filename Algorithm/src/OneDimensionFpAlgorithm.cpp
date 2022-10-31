#include "OneDimensionFpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "OneDimensionFpResults.hpp"
#include "OneDimensionFpCpuSimulation.hpp"
#if GPU_ENABLED == 1
#include "OneDimensionFpGpuSimulation.hpp"
#endif

void OneDimensionFpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
		OneDimensionFpGpuSimulation *oneDimensionFpGpuSimulation = new OneDimensionFpGpuSimulation();
		oneDimensionFpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif		
	}
	else
	{
		OneDimensionFpCpuSimulation *oneDimensionFpCpuSimulation = new OneDimensionFpCpuSimulation();
		oneDimensionFpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new OneDimensionFpResults();
	result->runAlgorithm(singleTone);

	if (singleTone->getInt("remove_log_files_after_simulation", 1))
	{
		unlink("log.dat");
	}
}