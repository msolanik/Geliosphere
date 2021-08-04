/**
 * @file CosmicFactory.hpp
 * @author Michal Solanik
 * @brief Implementation of factory pattern for Cosmic algorithms
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef COSMIC_FACTORY_H
#define COSMIC_FACTORY_H

#include "AbstractAlgorithmFactory.hpp"

/**
 * @brief Class represents implementation of Factory Pattern for 
 * cosmic algorithms.
 * 
 */
class CosmicFactory : public AbstractAlgorithmFactory {
public: 
	/**
	 * @brief Get the Algorithm object
	 * 
	 * @param name Name of algorithm
	 * @return AbstractAlgorithm* Created algorithm by given name.
	 */
	AbstractAlgorithm* getAlgorithm(std::string name); 
};

#endif // !COSMIC_FACTORY_H
