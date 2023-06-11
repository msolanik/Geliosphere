#include <string>
#include <regex>
#include <unistd.h>

#include "CLI/App.hpp"
#include "CLI/Option.hpp"
#include "spdlog/spdlog.h"

#include "ParseParams.hpp"
#include "ParamsCarrier.hpp"
#include "MeasureValuesTransformation.hpp"
#include "TomlSettings.hpp"

int ParseParams::parseParams(int argc, char **argv)
{
	std::string currentApplicationPath = getApplicationPath(argv);
	float newDT, newK, newV, newKparKper, newMu;
	int month, year;
	std::string newDestination, settings, customModelString;
	int bilions;
	singleTone = singleTone->instance();
	CLI::App app{"App description"};
	CLI::Option *forwardModel = app.add_flag("-F,--forward", "Run a 1D forward-in-time model")->group("models");
	CLI::Option *backwardModel = app.add_flag("-B,--backward", "Run a 1D backward-in-time model")->group("models");
	CLI::Option *solarPropLikeModel = app.add_flag("-E,--solarprop-like-model", "Run a SolarProp-like 2D backward model")->group("models");
	CLI::Option *geliosphereModel = app.add_flag("-T,--geliosphere-2d-model", "Run a Geliosphere 2D backward model")->group("models");
	CLI::Option *csv = app.add_flag("-c,--csv", "Output will be in .csv");
#if GPU_ENABLED == 1
	CLI::Option *cpuOnly = app.add_flag("--cpu-only", "Use only CPU for calculaions");
#else
	singleTone->putInt("isCpu", 1);
#endif		
	CLI::Option *dtset = app.add_option("-d,--dt", newDT, "Set dt to new value(s)");
	CLI::Option *kset = app.add_option("-K,--K0", newK, "Set K to new value(cm^2/s)");
	CLI::Option *vset = app.add_option("-V,--V", newV, "Set V to new value(km/s)");
	CLI::Option *destination = app.add_option("-p,--path", newDestination, "Set destination folder name");
	CLI::Option *setBilions = app.add_option("-N,--number-of-test-particles", bilions, "Set number of test particles in millions(round up due to GPU execution)");
	CLI::Option *monthOption = app.add_option("-m,--month", month, "Set month for using meassured values");
	CLI::Option *yearOption = app.add_option("-y,--year", year, "Set year for using meassured values");
	CLI::Option *settingsOption = app.add_option("-s,--settings", settings, "Path to .toml file");
	CLI::Option *customModel = app.add_option("--custom-model", customModelString, "Run custom user-implemented model.");
	
	kset->excludes(monthOption);
	kset->excludes(yearOption);

	vset->excludes(monthOption);
	vset->excludes(yearOption);

	backwardModel->excludes(forwardModel);
	backwardModel->excludes(solarPropLikeModel);
	backwardModel->excludes(geliosphereModel);
	backwardModel->excludes(customModel);
	forwardModel->excludes(backwardModel);
	forwardModel->excludes(solarPropLikeModel);
	forwardModel->excludes(geliosphereModel);
	forwardModel->excludes(customModel);
	solarPropLikeModel->excludes(backwardModel);
	solarPropLikeModel->excludes(forwardModel);
	solarPropLikeModel->excludes(geliosphereModel);
	solarPropLikeModel->excludes(customModel);
	geliosphereModel->excludes(backwardModel);
	geliosphereModel->excludes(forwardModel);
	geliosphereModel->excludes(solarPropLikeModel);
	geliosphereModel->excludes(customModel);
	customModel->excludes(backwardModel);
	customModel->excludes(forwardModel);
	customModel->excludes(solarPropLikeModel);
	customModel->excludes(geliosphereModel);

	monthOption->requires(yearOption);

	spdlog::info("Started to parsing input parameters");
	CLI11_PARSE(app, argc, argv);
	if (!*forwardModel && !*backwardModel && !*solarPropLikeModel && !geliosphereModel)
	{
		spdlog::error("At least one model must be selected!");
		return -1;
	}
	if (*csv)
	{
		singleTone->putInt("csv", 1);
	}
	if (*dtset)
	{
		if (newDT < 3.0 || newDT > 5000.0)
		{
			spdlog::error("dt is out of range!(3-5000)");
			return -1;
		}
		singleTone->putFloat("dt", newDT);
	}
	if (*kset)
	{
		if (newK < 0.0)
		{
			spdlog::error("K0 is out of range!(>0)");
			return -1;
		}
		if (newK < 1e19 || newK > 1e23)
		{
			spdlog::warn("K0 is out of recommended range!(1e19-1e23 cm^2/s)");
		}
		char buffer[80];
		sprintf(buffer, "%g", newK);
		singleTone->putString("K0input", buffer);
		// 10^22 cm^2/s = 4.444e-5 AU^2/s
		// 10^20 cm^2/s = 4.444e-7 AU^2/s
		newK = newK * 4.4683705e-27;
		singleTone->putFloat("K0", newK);
		singleTone->putInt("K0_entered_by_user", 1);
	}
	if (*setBilions)
	{
		if (bilions <= 0)
		{
			spdlog::error("Number of test particles must be greater than 0!");
			return -1;
		}
		singleTone->putInt("millions", bilions);
	}
	if (*vset)
	{
		if (newV < 100 || newV > 1500)
		{
			spdlog::error("V is out of range!(100-1500 km/s)");
			return -1;
		}
		singleTone->putString("Vinput", std::to_string(newV));
		newV = newV * 6.68458712e-9;
		singleTone->putFloat("V", newV);
	}

	if (*settingsOption)
	{
		if (access(settings.c_str(), F_OK) == 0) {
			TomlSettings *tomlSettings = new TomlSettings(settings);
			tomlSettings->parseFromSettings(singleTone);
		} else {
			spdlog::warn("No settings file exists on entered path.");
		}
	}
	else 
	{
		if (access(settings.c_str(), F_OK) == 0) {
			TomlSettings *tomlSettings = new TomlSettings(currentApplicationPath + "Settings.toml");
			tomlSettings->parseFromSettings(singleTone);
		} else {
			spdlog::warn("No settings file exists on default path.");
		}
	}
#if GPU_ENABLED == 1
	if (*cpuOnly)
	{
		singleTone->putInt("isCpu", 1);
	}
#endif
	if (*destination)
	{
		singleTone->putString("destination", newDestination);
	}
	if (*forwardModel)
	{
		singleTone->putString("model", "1D Fp");
	}
	else if (*backwardModel)
	{
		singleTone->putString("model", "1D Bp");
	}
	else if (*solarPropLikeModel)
	{
		singleTone->putString("model", "2D SolarProp-like");
	}
	else if (*geliosphereModel)
	{
		singleTone->putString("model", "2D Geliosphere");
	}
	else if (*customModel)
	{
		singleTone->putString("model", customModelString);
	}

	if (*monthOption && *yearOption)
	{
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

	printParameters(singleTone);
	return 1;
}

ParamsCarrier *ParseParams::getParams()
{
	return singleTone;
}

void ParseParams::printParameters(ParamsCarrier *params) 
{
	spdlog::info("Chosen model:" + singleTone->getString("model", "1D Fp"));
	spdlog::info("K0:" + std::to_string(params->getFloat("K0", params->getFloat("K0_default", 5e22 * 4.4683705e-27))) + " au^2 / s");
	spdlog::info("V:" + std::to_string(params->getFloat("V", params->getFloat("V_default", 400 * 6.68458712e-9))) + " au / s");
	spdlog::info("dt:" + std::to_string(params->getFloat("dt", params->getFloat("dt_default", 5.0f))) + " s");
	if (isInputSolarPropLikeModel(singleTone->getString("model", "1D Fp")) || isInputGeliosphere2DModel(singleTone->getString("model", "1D Fp")))
	{
		spdlog::info("tilt_angle:" + std::to_string(params->getFloat("tilt_angle", -1.0f)));
		spdlog::info("polarity:" + std::to_string(params->getInt("polarity", -1.0f)));
	}
}

std::string ParseParams::getTransformationTableName(std::string modelName)
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

bool ParseParams::isInputSolarPropLikeModel(std::string modelName)
{
	if (modelName.compare("2D SolarProp-like") == 0)
	{
		return true;
	}
	return false;
}

bool ParseParams::isInputGeliosphere2DModel(std::string modelName)
{
	if (modelName.compare("2D Geliosphere") == 0)
	{
		return true;
	}
	return false;
}

std::string ParseParams::getApplicationPath(char **argv)
{
	std::regex regexp(R"(.*\/)"); 
    std::cmatch m; 
    std::regex_search(argv[0], m, regexp); 
    return m[0]; 
}