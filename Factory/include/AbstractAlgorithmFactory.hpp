/**
 * @file AbstractAlgorithmFactory.hpp
 * @author Michal Solanik
 * @brief Interface of Abstract Factory Pattern
 * @version 0.1
 * @date 2021-07-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef ABSTRACT_ALGORITHM_FACTORY_H
#define ABSTRACT_ALGORITHM_FACTORY_H

#include <string>

#include "AbstractAlgorithm.hpp"
#include "InteractiveMode.hpp"

/**
 * @brief Interface representing functionality to implement Factory Pattern.
 * 
 */
class AbstractAlgorithmFactory {
public: 

	/**
	 * @brief Enum for defining type of factory
	 * 
	 */
	enum TYPE_ALGORITHM{
		COSMIC
	};

	/**
	 * @brief Get the Algorithm object
	 * 
	 * @param name 
	 * @return AbstractAlgorithm* 
	 */
	virtual AbstractAlgorithm* getAlgorithm(std::string name, InteractiveMode* interactiveMode) = 0; 

	/**
	 * @brief Create a Factory object
	 * 
	 * @param factory Type of factory
	 * @return AbstractAlgorithmFactory* Created factory by given type.
	 */
	static AbstractAlgorithmFactory* CreateFactory(TYPE_ALGORITHM factory);
};

#endif // !ABSTRACT_ALGORITHM_FACTORY
