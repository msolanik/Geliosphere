#include "OneDimensionBpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "OneDimensionBpResults.hpp"
#include "OneDimensionBpCpuModel.hpp"
#if GPU_ENABLED == 1
#include "OneDimensionBpGpuModel.hpp"
#endif

void OneDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (singleTone->getInt("evaluation",1))
	{
		if (!singleTone->getInt("isCpu", 0))
		{
	#if GPU_ENABLED == 1
			OneDimensionBpGpuModel *oneDimensionBpGpuSimulation = new OneDimensionBpGpuModel();
			oneDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
	#else
			spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
			return;
	#endif		
		}
		else
		{
			OneDimensionBpCpuModel *oneDimensionBpCpuSimulation = new OneDimensionBpCpuModel();
			oneDimensionBpCpuSimulation->runSimulation(singleTone);
		}
	}
	
	AbstractAlgorithm *result;
	result = new OneDimensionBpResults();
	result->runAlgorithm(singleTone);
	

	if (singleTone->getInt("remove_log_files_after_simulation", 1))
	{
		unlink("log.dat");
	}
}