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
#include "InputValidation.hpp"

int ParseParams::parseParams(int argc, char **argv)
{
	std::string inputFile;
	float newDt, newK0, newV, newMu;
	int month, year;
	std::string newDestination, settings, customModelString;
	int numberOfTestParticles;
	singleTone = singleTone->instance();
	std::string currentApplicationPath = getApplicationPath(argv);
	singleTone->putString("currentApplicationPath", currentApplicationPath);
	InputValidation *inputValidation = new InputValidation();
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
	CLI::Option *batchRun = app.add_option("-b,--batchrun", inputFile,"Input batch file")->group("models");
	CLI::Option *dtset = app.add_option("-d,--dt", newDt, "Set dt to new value(s)");
	CLI::Option *kset = app.add_option("-K,--K0", newK0, "Set K to new value(cm^2/s)");
	CLI::Option *vset = app.add_option("-V,--V", newV, "Set V to new value(km/s)");
	CLI::Option *destination = app.add_option("-p,--path", newDestination, "Set destination folder name");
	CLI::Option *setNumberOfTestParticles = app.add_option("-N,--number-of-test-particles", numberOfTestParticles, "Set number of test particles in millions(round up due to GPU execution)");
	CLI::Option *monthOption = app.add_option("-m,--month", month, "Set month for using meassured values");
	CLI::Option *yearOption = app.add_option("-y,--year", year, "Set year for using meassured values");
	CLI::Option *settingsOption = app.add_option("-s,--settings", settings, "Path to .toml file");
	CLI::Option *customModel = app.add_option("--custom-model", customModelString, "Run custom user-implemented model.");

	kset->excludes(monthOption, yearOption);
	vset->excludes(monthOption, yearOption);

	backwardModel->excludes(forwardModel, solarPropLikeModel, geliosphereModel, customModel, batchRun);
	forwardModel->excludes(backwardModel, solarPropLikeModel, geliosphereModel, customModel, batchRun);
	solarPropLikeModel->excludes(backwardModel, forwardModel, geliosphereModel, customModel, batchRun);
	geliosphereModel->excludes(backwardModel, forwardModel, solarPropLikeModel, customModel, batchRun);
	customModel->excludes(backwardModel, forwardModel, solarPropLikeModel, geliosphereModel, batchRun);
	batchRun->excludes(backwardModel, forwardModel, solarPropLikeModel, geliosphereModel, customModel,
		dtset, setNumberOfTestParticles, kset, vset, monthOption, yearOption);

	monthOption->requires(yearOption);

	spdlog::info("Started to parsing input parameters");
	CLI11_PARSE(app, argc, argv);
	if (!*forwardModel && !*backwardModel && !*solarPropLikeModel && !geliosphereModel && !batchRun)
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
		if (!inputValidation->checkDt(newDt))
		{
			spdlog::error("dt is out of range!(3-5000)");
			return false;
		}
		inputValidation->setDt(singleTone, newDt);
	}
	if (*kset)
	{
		if (!inputValidation->checkK0(newK0))
		{
			spdlog::error("K0 is out of range!(>0)");
			return -1;
		}
		if (newK0 < 1e19 || newK0 > 1e23)
		{
			spdlog::warn("K0 is out of recommended range!(1e19-1e23 cm^2/s)");
		}
		inputValidation->setK0(singleTone, newK0);
	}
	if (*setNumberOfTestParticles)
	{
		if (!inputValidation->checkNumberOfTestParticles(numberOfTestParticles))
		{
			spdlog::error("Number of test particles must be greater than 0!");
			return -1;
		}
		inputValidation->setNumberOfTestParticles(singleTone, numberOfTestParticles);
	}
	if (*vset)
	{
		if (!inputValidation->checkV(newV))
		{
			spdlog::error("V is out of range!(100-1500 km/s)");
			return -1;
		}
		inputValidation->setV(singleTone, newV);
	}
	if (*settingsOption)
	{
		inputValidation->newSettingsLocationCheck(singleTone, settings);
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
	else if (*batchRun)
	{
		singleTone->putString("model", "batch run");
		singleTone->putString("inputBatchFile", inputFile);
		return 1;
	}
	if (*monthOption && *yearOption)
	{
		inputValidation->monthYearCheck(singleTone, year, month, currentApplicationPath);
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
	InputValidation *inputValidation = new InputValidation();
	spdlog::info("Chosen model:" + singleTone->getString("model", "1D Fp"));
	spdlog::info("K0:" + std::to_string(params->getFloat("K0", params->getFloat("K0_default", 5e22 * 4.4683705e-27))) + " au^2 / s");
	spdlog::info("V:" + std::to_string(params->getFloat("V", params->getFloat("V_default", 400 * 6.68458712e-9))) + " au / s");
	spdlog::info("dt:" + std::to_string(params->getFloat("dt", params->getFloat("dt_default", 5.0f))) + " s");
	if (inputValidation->isInputSolarPropLikeModel(singleTone->getString("model", "1D Fp")) || inputValidation->isInputGeliosphere2DModel(singleTone->getString("model", "1D Fp")))
	{
		spdlog::info("tilt_angle:" + std::to_string(params->getFloat("tilt_angle", -1.0f)));
		spdlog::info("polarity:" + std::to_string(params->getInt("polarity", -1.0f)));
	}
}

std::string ParseParams::getApplicationPath(char **argv)
{
	std::regex regexp(R"(.*\/)"); 
    std::cmatch m; 
    std::regex_search(argv[0], m, regexp); 
    return m[0]; 
}