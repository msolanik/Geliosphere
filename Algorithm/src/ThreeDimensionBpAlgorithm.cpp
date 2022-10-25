#include "ThreeDimensionBpAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "ThreeDimensionBpCpuSimulation.hpp"
#include "TwoDimensionBpResults.hpp"
#if GPU_ENABLED == 1
#include "ThreeDimensionBpGpuSimulation.hpp"
#endif

void ThreeDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
		ThreeDimensionBpGpuSimulation *threeDimensionBpGpuSimulation = new ThreeDimensionBpGpuSimulation();
		threeDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif	
	}
	else
	{
		ThreeDimensionBpCpuSimulation *threeDimensionBpCpuSimulation = new ThreeDimensionBpCpuSimulation();
		threeDimensionBpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);

	if (singleTone->getInt("remove_log_files_after_simulation", 1))
	{
		unlink("log.dat");
	}
}