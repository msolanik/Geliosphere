
#include "InputValidation.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "CLI/App.hpp"
#include "CLI/Option.hpp"
#include "spdlog/spdlog.h"

#include "MeasureValuesTransformation.hpp"
#include "ParamsCarrier.hpp"
#include "ParseParams.hpp"
#include "TomlSettings.hpp"

std::string InputValidation::getTransformationTableName(std::string modelName)
{
	if (isInputSolarPropLikeModel(modelName))
	{
		return "SolarProp_K0_phi_table.csv";
	}
	else if (isInputGeliosphere2DModel(modelName))
	{
		return "Geliosphere_K0_phi_table.csv";
	}
	else 
	{
		return "K0_phi_table.csv";
	}
	return NULL;
}

bool InputValidation::isInputSolarPropLikeModel(std::string modelName)
{
	if (modelName.compare("2D SolarProp-like") == 0)
	{
		return true;
	}
	return false;
}

bool InputValidation::isInputGeliosphere2DModel(std::string modelName)
{
	if (modelName.compare("2D Geliosphere") == 0)
	{
		return true;
	}
	return false;
}

void InputValidation::setDt(ParamsCarrier *singleTone, float dt)
{
	singleTone->putFloat("dt", dt);
}

bool InputValidation::checkDt(float dt)
{
	if (dt < 3.0 || dt > 5000.0)
	{
		return false;
	}
	return true;
}

void InputValidation::setK0(ParamsCarrier *singleTone, float K0)
{
	char buffer[80];
	sprintf(buffer, "%g", K0);
	singleTone->putString("K0input", buffer);
	float newK = K0 * 4.4683705e-27;
	singleTone->putFloat("K0", newK);
	singleTone->putInt("K0_entered_by_user", 1);
}

bool InputValidation::checkK0(float K0)
{
	if (K0 < 0.0)
	{
		return false;
	}
	return true;
}

void InputValidation::setV(ParamsCarrier *singleTone, float V)
{
	singleTone->putString("Vinput", std::to_string(V));
	float newV = V * 6.68458712e-9;
	singleTone->putFloat("V", newV);
}
	
bool InputValidation::checkV(float V)
{
	if (V < 100.0 || V > 1500.0)
	{
		return false;
	}
	return true;
}

void InputValidation::newSettingsLocationCheck(ParamsCarrier *singleTone, std::string settings){
	if (access(settings.c_str(), F_OK) == 0) {
		TomlSettings *tomlSettings = new TomlSettings(settings);
		tomlSettings->parseFromSettings(singleTone);
	} else {
		spdlog::warn("No settings file exists on entered path.");
	}
}

void InputValidation::setNumberOfTestParticles(ParamsCarrier *singleTone, int numberOfTestParticles)
{
	singleTone->putInt("millions", numberOfTestParticles);
}

bool InputValidation::checkNumberOfTestParticles(int numberOfTestParticles)
{
	if (numberOfTestParticles <= 0)
	{
		return false;
	}
	return true;
}

void InputValidation::monthYearCheck(ParamsCarrier *singleTone, int year, int month, std::string currentApplicationPath){
	try
		{
			singleTone->putInt("month_option", month);
			singleTone->putInt("year_option", year);
			MeasureValuesTransformation *measureValuesTransformation = new MeasureValuesTransformation(
				currentApplicationPath + getTransformationTableName(singleTone->getString("model", "1D Fp")), singleTone->getString("model", "1D Fp"));
			singleTone->putFloat("K0", measureValuesTransformation->getDiffusionCoefficientValue(month, year));
			if (isInputSolarPropLikeModel(singleTone->getString("model", "1D Fp")) || isInputGeliosphere2DModel(singleTone->getString("model", "1D Fp")))
			{
				singleTone->putFloat("tilt_angle", measureValuesTransformation->getTiltAngle(month, year));
				singleTone->putInt("polarity", measureValuesTransformation->getPolarity(month, year));
			}
			else
			{
				singleTone->putFloat("V", measureValuesTransformation->getSolarWindSpeedValue(month, year) * 6.68458712e-9);
			}
		}
		catch(const std::out_of_range&)
		{
			spdlog::error("Combination for entered date was not found in input table.");
		}
}


