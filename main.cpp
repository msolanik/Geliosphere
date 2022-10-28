/**
 * @file main.cpp
 * @author Michal Solanik
 * @brief Main function
 * @version 0.1
 * @date 2021-07-09
 * 
 * @details Main function is used to parse arguments and
 * 	start new simulation with given parameters.  
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>

#include "ParamsCarrier.hpp"
#include "ParseParams.hpp"
#include "AbstractAlgorithm.hpp"
#include "AbstractAlgorithmFactory.hpp"

int main(int argc, char **argv)
{
	AbstractAlgorithmFactory *factory = AbstractAlgorithmFactory::CreateFactory(AbstractAlgorithmFactory::TYPE_ALGORITHM::COSMIC);
	ParseParams *parse = new ParseParams();
	if (parse->parseParams(argc, argv) != 1)
	{
		return -1;
	}
	ParamsCarrier *singleTone;
	singleTone = parse->getParams();
	AbstractAlgorithm *actualAlgorithm;
	actualAlgorithm = factory->getAlgorithm(singleTone->getString("model", "FWMethod"));
	actualAlgorithm->runAlgorithm(singleTone);
	return 0;
}