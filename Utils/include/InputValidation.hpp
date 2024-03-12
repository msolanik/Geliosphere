/**
 * @file InputValidation.hpp
 * @author Tomas Telepcak
 * @brief Utilities for validating input data, creating input settings
 * @version 0.1
 * @date 2024-01-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef INPUT_VALIDATION
#define INPUT_VALIDATION


#include "ParamsCarrier.hpp"

class InputValidation
{
public:
	void generateTomlFile(double r, double theta);
	void newSettingsLocationCheck(ParamsCarrier *singleTone, std::string settings);
	void dtSetCheck(ParamsCarrier *singleTone, float newDT);
	void monthYearCheck(ParamsCarrier *singleTone, int year, int month,std::string currentApplicationPath);
	std::string getTransformationTableName(std::string modelName);
	bool isInputSolarPropLikeModel(std::string modelName);
	bool isInputGeliosphere2DModel(std::string modelName);
};


#endif 