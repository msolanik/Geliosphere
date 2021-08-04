#include "AbstractAlgorithmFactory.hpp"
#include "CosmicFactory.hpp"

AbstractAlgorithmFactory* AbstractAlgorithmFactory::CreateFactory(TYPE_ALGORITHM factory) {
	if (factory == TYPE_ALGORITHM::COSMIC) {
		return new CosmicFactory(); 
	}
	return NULL; 
}