#include "CosmicFactory.hpp"
#include "OneDimensionBpAlgorithm.hpp"
#include "OneDimensionFpAlgorithm.hpp"
#include "TwoDimensionBpAlgorithm.hpp"
#include "ThreeDimensionBpAlgorithm.hpp"

AbstractAlgorithm *CosmicFactory::getAlgorithm(std::string name)
{
	if (name.compare("FWMethod") == 0)
	{
		return new OneDimensionFpAlgorithm();
	}
	else if (name.compare("BPMethod") == 0)
	{
		return new OneDimensionBpAlgorithm();
	}
	else if (name.compare("TwoDimensionBp") == 0)
	{
		return new TwoDimensionBpAlgorithm();
	}
	else if (name.compare("ThreeDimensionBp") == 0)
	{
		return new ThreeDimensionBpAlgorithm();
	}
	return NULL;
}