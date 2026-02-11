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
	ParseConfig config;
	singleTone = singleTone->instance();
	config.currentApplicationPath = getApplicationPath(argv);
	singleTone->putString("currentApplicationPath", config.currentApplicationPath);
	
	CLI::App app{"App description"};
	
	// Initialize state machine
	ParseState currentState = ParseState::SETUP;
	
	// Run state machine
	while (currentState != ParseState::COMPLETE && currentState != ParseState::ERROR) {
		currentState = executeState(currentState, config, app, argc, argv);
	}
	
	if (currentState == ParseState::ERROR) {
		return -1;
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

ParseParams::ParseState ParseParams::executeState(ParseState currentState, ParseConfig& config, CLI::App& app, int argc, char** argv)
{
	switch (currentState) {
		case ParseState::SETUP:
			return handleSetupState(config, app);
		case ParseState::PARSING:
			return handleParsingState(app, argc, argv);
		case ParseState::VALIDATION:
			return handleValidationState();
		case ParseState::PROCESSING:
			return handleProcessingState(config);
		default:
			return ParseState::ERROR;
	}
}

ParseParams::ParseState ParseParams::handleSetupState(ParseConfig& config, CLI::App& app)
{
	// Setup CLI options
	setupCliOptions(app, config);
	
	// Setup option relationships
	setupOptionRelationships();
	
	return ParseState::PARSING;
}

ParseParams::ParseState ParseParams::handleParsingState(CLI::App& app, int argc, char** argv)
{
	spdlog::info("Started to parsing input parameters");
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		app.exit(e);
		return ParseState::ERROR;
	}
	
	return ParseState::VALIDATION;
}

ParseParams::ParseState ParseParams::handleValidationState()
{
	// Validate that at least one model is selected
	if (!*forwardModel && !*backwardModel && !*solarPropLikeModel && !*geliosphereModel && !*batchRun) {
		spdlog::error("At least one model must be selected!");
		return ParseState::ERROR;
	}
	
	return ParseState::PROCESSING;
}

ParseParams::ParseState ParseParams::handleProcessingState(const ParseConfig& config)
{
	InputValidation *inputValidation = new InputValidation();
	
	// Process general options
	if (*run_simulation) {
		singleTone->putInt("run_simulation", 0);
		singleTone->putString("pathToLogFile", config.pathToLogFile);
	} else {
		singleTone->putInt("run_simulation", 1);
	}
	
	if (*csv) {
		singleTone->putInt("csv", 1);
	}
	
#if GPU_ENABLED == 1
	if (*cpuOnly) {
		singleTone->putInt("isCpu", 1);
	}
#endif

	if (*destination) {
		singleTone->putString("destination", config.newDestination);
	}
	
	// Process value options with validation
	if (*dtset) {
		if (!inputValidation->checkDt(config.newDt)) {
			spdlog::error("dt is out of range!(3-5000)");
			return ParseState::ERROR;
		}
		inputValidation->setDt(singleTone, config.newDt);
	}
	
	if (*kset) {
		if (!inputValidation->checkK0(config.newK0)) {
			spdlog::error("K0 is out of range!(>0)");
			return ParseState::ERROR;
		}
		if (config.newK0 < 1e19 || config.newK0 > 1e23) {
			spdlog::warn("K0 is out of recommended range!(1e19-1e23 cm^2/s)");
		}
		inputValidation->setK0(singleTone, config.newK0);
	}
	
	if (*setNumberOfTestParticles) {
		if (!inputValidation->checkNumberOfTestParticles(config.numberOfTestParticles)) {
			spdlog::error("Number of test particles must be greater than 0!");
			return ParseState::ERROR;
		}
		inputValidation->setNumberOfTestParticles(singleTone, config.numberOfTestParticles);
	}
	
	if (*vset) {
		if (!inputValidation->checkV(config.newV)) {
			spdlog::error("V is out of range!(100-1500 km/s)");
			return ParseState::ERROR;
		}
		inputValidation->setV(singleTone, config.newV);
	}
	
	// Process model options
	if (*forwardModel) {
		singleTone->putString("model", "1D Fp");
	} else if (*backwardModel) {
		singleTone->putString("model", "1D Bp");
	} else if (*solarPropLikeModel) {
		singleTone->putString("model", "2D SolarProp-like");
	} else if (*geliosphereModel) {
		singleTone->putString("model", "2D Geliosphere");
	} else if (*customModel) {
		singleTone->putString("model", config.customModelString);
	} else if (*batchRun) {
		singleTone->putString("model", "batch run");
		singleTone->putString("inputBatchFile", config.inputFile);
		return ParseState::COMPLETE; // Special case - batch run returns early
	}
	
	// Process settings options
	if (*settingsOption) {
		inputValidation->newSettingsLocationCheck(singleTone, config.settings);
	} else {
		if (access(config.settings.c_str(), F_OK) == 0) {
			TomlSettings *tomlSettings = new TomlSettings(config.currentApplicationPath + "Settings.toml");
			tomlSettings->parseFromSettings(singleTone);
		} else {
			spdlog::warn("No settings file exists on default path.");
		}
	}
	
	if (*monthOption && *yearOption) {
		inputValidation->monthYearCheck(singleTone, config.year, config.month, config.currentApplicationPath);
	}
	
	return ParseState::COMPLETE;
}
void ParseParams::setupCliOptions(CLI::App& app, ParseConfig& config)
{
	forwardModel = app.add_flag("-F,--forward", "Run a 1D forward-in-time model")->group("models");
	backwardModel = app.add_flag("-B,--backward", "Run a 1D backward-in-time model")->group("models");
	solarPropLikeModel = app.add_flag("-E,--solarprop-like-model", "Run a SolarProp-like 2D backward model")->group("models");
	geliosphereModel = app.add_flag("-T,--geliosphere-2d-model", "Run a Geliosphere 2D backward model")->group("models");
	csv = app.add_flag("-c,--csv", "Output will be in .csv");
	run_simulation = app.add_option("--evaluation", config.pathToLogFile, "Simulation excluded, run only evaluation ");
#if GPU_ENABLED == 1
	cpuOnly = app.add_flag("--cpu-only", "Use only CPU for calculaions");
#else
	singleTone->putInt("isCpu", 1);
#endif		
	batchRun = app.add_option("-b,--batchrun", config.inputFile, "Input batch file")->group("models");
	dtset = app.add_option("-d,--dt", config.newDt, "Set dt to new value(s)");
	kset = app.add_option("-K,--K0", config.newK0, "Set K to new value(cm^2/s)");
	vset = app.add_option("-V,--V", config.newV, "Set V to new value(km/s)");
	destination = app.add_option("-p,--path", config.newDestination, "Set destination folder name");
	setNumberOfTestParticles = app.add_option("-N,--number-of-test-particles", config.numberOfTestParticles, "Set number of test particles in millions(round up due to GPU execution)");
	monthOption = app.add_option("-m,--month", config.month, "Set month for using meassured values");
	yearOption = app.add_option("-y,--year", config.year, "Set year for using meassured values");
	settingsOption = app.add_option("-s,--settings", config.settings, "Path to .toml file");
	customModel = app.add_option("--custom-model", config.customModelString, "Run custom user-implemented model.");
}

void ParseParams::setupOptionRelationships()
{
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
}