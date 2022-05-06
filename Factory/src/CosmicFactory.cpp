#include "CosmicFactory.hpp"
#include "OneDimensionBpAlgorithm.hpp"
#include "OneDimensionFpAlgorithm.hpp"
#include "TwoDimensionBpAlgorithm.hpp"

AbstractAlgorithm* CosmicFactory::getAlgorithm(std::string name, InteractiveMode* interactiveMode) {
	if (name.compare("FWMethod") == 0) {
		return new OneDimensionFpAlgorithm(interactiveMode); 
	} else if (name.compare("BPMethod") == 0) {
		return new OneDimensionBpAlgorithm(interactiveMode);
	} else if (name.compare("TwoDimensionBp") == 0) {
		return new TwoDimensionBpAlgorithm();
	}
	return NULL; 
}