#include "CosmicFactory.hpp"
#include "OneDimensionBpAlgorithm.hpp"
#include "OneDimensionFpAlgorithm.hpp"
#include "SolarPropLikeAlgorithm.hpp"
#include "GeliosphereAlgorithm.hpp"

AbstractAlgorithm *CosmicFactory::getAlgorithm(std::string name)
{
	if (name.compare("1D Fp") == 0)
	{
		return new OneDimensionFpAlgorithm();
	}
	else if (name.compare("1D Bp") == 0)
	{
		return new OneDimensionBpAlgorithm();
	}
	else if (name.compare("2D SolarProp-like") == 0)
	{
		return new SolarPropLikeAlgorithm();
	}
	else if (name.compare("2D Geliosphere") == 0)
	{
		return new GeliosphereAlgorithm();
	}
	return NULL;
}