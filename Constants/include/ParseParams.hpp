/**
 * @file ParseParams.hpp
 * @author Michal Solanik
 * @brief Parser of arguements from CLI
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef PARSE_PARAMS_H
#define PARSE_PARAMS_H

#include <string>
#include "ParamsCarrier.hpp"

/**
 * @brief ParseParams is responsible for parsing arguments from CLI.
 * 
 */
class ParseParams
{
public:
	/**
	 * @brief Parse params from CLI.
	 * 
	 * @param argc Number of arguments
	 * @param argv Arguments
	 * @return 1 in the case of successfully parsed arguments and
	 * -1 in the case of failure. 
	 */
	int parseParams(int argc, char **argv);

	/**
	 * @brief Get the Params object
	 * 
	 * @return ParamsCarrier* with parsed arguments
	 */
	ParamsCarrier *getParams();

private:
	/**
	 * @brief Instance of ParamsCarrier for placing 
	 * parsed arguments.
	 * 
	 */
	ParamsCarrier *singleTone;

	/**
	 * @brief Print basic parameters which are used in simulation.
	 * 
	 * @param params Data structure holding input data.
	 */
	void printParameters(ParamsCarrier *params);

	/**
	 * @brief Get name for transformation table.
	 * 
	 * @param params Data structure holding input data.
	 * @return Name of the file containing transformation table. 
	 */
	std::string getTransformationTableName(std::string modelName);

	/**
	 * @brief Return true for 2D models.
	 * 
	 * @param modelName Name of the model.
	 */
	bool isInput2DModel(std::string modelName);
};

#endif