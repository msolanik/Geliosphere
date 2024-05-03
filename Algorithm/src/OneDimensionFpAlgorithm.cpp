#include "OneDimensionFpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "OneDimensionFpResults.hpp"
#include "OneDimensionFpCpuModel.hpp"
#if GPU_ENABLED == 1
#include "OneDimensionFpGpuModel.hpp"
#endif

void OneDimensionFpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
		OneDimensionFpGpuModel *oneDimensionFpGpuSimulation = new OneDimensionFpGpuModel();
		oneDimensionFpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif		
	}
	else
	{
		OneDimensionFpCpuModel *oneDimensionFpCpuSimulation = new OneDimensionFpCpuModel();
		oneDimensionFpCpuSimulation->runSimulation(singleTone);
	}

	if (singleTone->getInt("evaluation",0))
	{
		AbstractAlgorithm *result;
		result = new OneDimensionFpResults();
		result->runAlgorithm(singleTone);
	}

	if (singleTone->getInt("remove_log_files_after_simulation", 1))
	{
		unlink("log.dat");
	}
}