#include "ThreeDimensionBpAlgorithm.hpp"

#include "ThreeDimensionBpCpuSimulation.hpp"
#include "TwoDimensionBpResults.hpp"

void ThreeDimensionBpAlgorithm::runAlgorithm(ParamsCarrier *singleTone)
{
	if (!singleTone->getInt("isCpu", 0))
	{

	}
	else
	{
		ThreeDimensionBpCpuSimulation *threeDimensionBpCpuSimulation = new ThreeDimensionBpCpuSimulation();
		threeDimensionBpCpuSimulation->runSimulation(singleTone);
	}

	AbstractAlgorithm *result;
	result = new TwoDimensionBpResults();
	result->runAlgorithm(singleTone);
}