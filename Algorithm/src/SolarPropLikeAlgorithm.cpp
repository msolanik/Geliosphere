#include "SolarPropLikeAlgorithm.hpp"

#include "spdlog/spdlog.h"

#include "TwoDimensionBpResults.hpp"
#include "SolarPropLikeCpuModel.hpp"
#if GPU_ENABLED == 1
#include "SolarPropLikeGpuModel.hpp"
#endif

void SolarPropLikeAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (singleTone->getInt("evaluation",1))
	{
		if (!singleTone->getInt("isCpu", 0))
		{
	#if GPU_ENABLED == 1
			SolarPropLikeGpuModel *twoDimensionBpGpuSimulation = new SolarPropLikeGpuModel();
			twoDimensionBpGpuSimulation->prepareAndRunSimulation(singleTone);
	#else
			spdlog::info("GPU-based computations are disabled. Please, compile again without -DUSE_CPU_ONLY.");
			return;
	#endif		
		}
		else
		{
			SolarPropLikeCpuModel *twoDimensionBpCpuSimulation = new SolarPropLikeCpuModel();
			twoDimensionBpCpuSimulation->runSimulation(singleTone);
		}
	}
	
	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);
	

	if (singleTone->getInt("remove_log_files_after_simulation", 1))
	{
		unlink("log.dat");
	}
}