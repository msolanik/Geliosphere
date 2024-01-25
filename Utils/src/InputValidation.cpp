
#include "InputValidation.hpp"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>

#include <string>
#include <regex>
#include <unistd.h>
#include <filesystem>

#include "CLI/App.hpp"
#include "CLI/Option.hpp"
#include "spdlog/spdlog.h"

#include "ParseParams.hpp"
#include "ParamsCarrier.hpp"
#include "MeasureValuesTransformation.hpp"
#include "TomlSettings.hpp"


using namespace std;
/*
void generateTomlFile(double r, double theta) {
    //std::cout << "r: " << r << " theta: " << theta << std::endl;
    auto tbl = toml::table{
        { "default_values", toml::table{
            { "uniform_energy_injection_maximum", 3.0 } ,
            { "K0", 5e+22 },
            { "V", 400.0 },
            { "dt", 50.0 },
            { "r_injection", r }} 
        },
        { "2d_models_common_settings", toml::table{
            { "theta_injection", theta } ,
            { "use_uniform_injection", true }}
        },
        { "SolarProp_like_model_settings", toml::table{
            { "SolarProp_ratio", 0.02 } } 
        },
        { "Geliosphere_model_settings", toml::table{
            { "Geliosphere_ratio", 0.2 } ,
            { "K0_ratio", 1.0 },
            { "C_delta", 8.7e-05 },
            { "default_tilt_angle", 0.1 }} 
        },
        { "advanced_settings", toml::table{
            { "remove_log_files_after_simulation", true } } 
        }
    };
    std::ofstream file("./Settings_batch.toml");
    
    if (file.is_open()) {
        file << tbl;
        file.close();
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }
    
}*/


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

void  InputValidation::dtSetCheck(ParamsCarrier *singleTone, float newDT ){
	if (newDT < 3.0 || newDT > 5000.0)
		{
			spdlog::error("dt is out of range!(3-5000)");
			return ;
		}
		singleTone->putFloat("dt", newDT);
}

void InputValidation::newSettingsLocationCheck(ParamsCarrier *singleTone, std::string settings){
	if (access(settings.c_str(), F_OK) == 0) {
		TomlSettings *tomlSettings = new TomlSettings(settings);
		tomlSettings->parseFromSettings(singleTone);
	} else {
		spdlog::warn(settings);
		spdlog::warn("No settings file exists on entered path.");
	}
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


