#include "GeliosphereAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "GeliosphereCpuModel.hpp"
#include "TwoDimensionBpResults.hpp"
#if GPU_ENABLED == 1
#include "GeliosphereGpuModel.hpp"
#endif

void GeliosphereAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{
#if GPU_ENABLED == 1
		GeliosphereGpuModel *threeDimensionBpGpuSimulation = new GeliosphereGpuModel();
		threeDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
#else
    	spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
		return;
#endif	
	}
	else
	{
		GeliosphereCpuModel *threeDimensionBpCpuSimulation = new GeliosphereCpuModel();
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