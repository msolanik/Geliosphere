/**
 * @file SolarPropLikeAlgorithm.hpp
 * @author Michal Solanik
 * @brief Implementation of Geliosphere model algorithm
 * @version 0.2
 * @date 2022-07-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef GELIOSPHERE_MODEL_H
#define GELIOSPHERE_MODEL_H

#include "AbstractAlgorithm.hpp"

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running implementation of Geliosphere model simulation.
 * 
 */
class GeliosphereAlgorithm : public AbstractAlgorithm
{
public:
	/**
	 * @brief GPU implementation of 2D B-p model and set input 
	 * paramaters.  
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif