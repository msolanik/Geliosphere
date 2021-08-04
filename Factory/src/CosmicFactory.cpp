#include "CosmicFactory.hpp"
#include "OneDimensionBpAlgorithm.hpp"
#include "OneDimensionFpAlgorithm.hpp"

AbstractAlgorithm* CosmicFactory::getAlgorithm(std::string name) {
	if (name.compare("FWMethod") == 0) {
		return new OneDimensionFpAlgorithm(); 
	} else if (name.compare("BPMethod") == 0) {
		return new OneDimensionBpAlgorithm();
	}
	return NULL; 
}