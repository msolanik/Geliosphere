#include <string>

#include "CLI/App.hpp"
#include "CLI/Option.hpp"
#include "spdlog/spdlog.h"

#include "ParseParams.hpp"
#include "ParamsCarrier.hpp"
#include "MeasureValuesTransformation.hpp"
#include "TomlSettings.hpp"

int ParseParams::parseParams(int argc, char **argv)
{
	float newDT, newK, newV, newKparKper, newMu;
	int month, year;
	std::string newDestination, settings;
	int bilions;
	singleTone = singleTone->instance();
	CLI::App app{"App description"};
	CLI::Option *forwardMethod = app.add_flag("-F,--Forward", "Run a forward method")->group("Methods");
	CLI::Option *backwardMethod = app.add_flag("-B,--Backward", "Run a backward method")->group("Methods");
	CLI::Option *twoDimensionBackwardMethod = app.add_flag("-E,--TwoDimensionBackward", "Run a 2D backward method")->group("Methods");
	CLI::Option *threeDimensionBackwardMethod = app.add_flag("-T,--ThreeDimensionBackward", "Run a 3D backward method")->group("Methods");
	CLI::Option *csv = app.add_flag("-c,--csv", "Output will be in .csv");
#if GPU_ENABLED == 1
	CLI::Option *cpuOnly = app.add_flag("--cpu_only", "Use only CPU for calculaions");
#else
	singleTone->putInt("isCpu", 1);
#endif		
	CLI::Option *dtset = app.add_option("-d,--dt", newDT, "Set dt to new value(s)");
	CLI::Option *kset = app.add_option("-K,--K0", newK, "Set K to new value(cm^2/s)");
	CLI::Option *vset = app.add_option("-V,--V", newV, "Set V to new value(km/s)");
	CLI::Option *destination = app.add_option("-p,--path", newDestination, "Set destination folder name");
	CLI::Option *setBilions = app.add_option("-N,--millions", bilions, "Set number of simulations in millions(round up due to GPU execution)");
	CLI::Option *monthOption = app.add_option("-m,--month", month, "Set month for using meassured values");
	CLI::Option *yearOption = app.add_option("-y,--year", year, "Set year for using meassured values");
	CLI::Option *settingsOption = app.add_option("-s,--settings", settings, "");
	
	kset->excludes(monthOption);
	kset->excludes(yearOption);

	vset->excludes(monthOption);
	vset->excludes(yearOption);

	backwardMethod->excludes(forwardMethod);
	backwardMethod->excludes(twoDimensionBackwardMethod);
	backwardMethod->excludes(threeDimensionBackwardMethod);
	forwardMethod->excludes(backwardMethod);
	forwardMethod->excludes(twoDimensionBackwardMethod);
	forwardMethod->excludes(threeDimensionBackwardMethod);
	twoDimensionBackwardMethod->excludes(backwardMethod);
	twoDimensionBackwardMethod->excludes(forwardMethod);
	twoDimensionBackwardMethod->excludes(threeDimensionBackwardMethod);
	threeDimensionBackwardMethod->excludes(backwardMethod);
	threeDimensionBackwardMethod->excludes(forwardMethod);
	threeDimensionBackwardMethod->excludes(twoDimensionBackwardMethod);


	monthOption->requires(yearOption);

	spdlog::info("Started to parsing input parameters");
	CLI11_PARSE(app, argc, argv);
	if (!*forwardMethod && !*backwardMethod && !*twoDimensionBackwardMethod && !threeDimensionBackwardMethod)
	{
		spdlog::error("At least one method must be selected!");
		return -1;
	}
	if (*csv)
	{
		singleTone->putInt("csv", 1);
	}
	if (*dtset)
	{
		if (newDT < 0.1f || newDT > 1000)
		{
			spdlog::error("dt is out of range!(0.1-1000)");
			return -1;
		}
		singleTone->putFloat("dt", newDT);
	}
	if (*kset)
	{
		if (newK < 1e22 || newK > 1e23)
		{
			spdlog::error("K0 is out of range!(1e22-1e23 cm^2/s)");
			return -1;
		}
		char buffer[80];
		sprintf(buffer, "%g", newK);
		singleTone->putString("K0input", buffer);
		newK = newK * 4.4683705e-27;
		singleTone->putFloat("K0", newK);
		singleTone->putInt("K0_entered_by_user", 1);
	}
	if (*setBilions)
	{
		if (bilions <= 0)
		{
			spdlog::error("Number of simulations must be greater than 0!");
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
		TomlSettings *tomlSettings = new TomlSettings(settings);
		tomlSettings->parseFromSettings(singleTone);
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
	if (*forwardMethod)
	{
		singleTone->putString("model", "FWMethod");
	}
	else if (*backwardMethod)
	{
		singleTone->putString("model", "BPMethod");
	}
	else if (*twoDimensionBackwardMethod)
	{
		singleTone->putString("model", "TwoDimensionBp");
	}
	else if (*threeDimensionBackwardMethod)
	{
		singleTone->putString("model", "ThreeDimensionBp");
	}

	if (*monthOption && *yearOption)
	{
		MeasureValuesTransformation *measureValuesTransformation = new MeasureValuesTransformation(
			getTransformationTableName(singleTone->getString("model", "FWMethod")));
		singleTone->putFloat("K0", measureValuesTransformation->getDiffusionCoefficientValue(month, year));
		singleTone->putFloat("V", measureValuesTransformation->getSolarWindSpeedValue(month, year));
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
	spdlog::info("Chosen model:" + singleTone->getString("model", "FWMethod"));
	spdlog::info("K0:" + std::to_string(params->getFloat("K0", params->getFloat("K0_default", 5e22 * 4.4683705e-27))) + " au^2 / s");
	spdlog::info("V:" + std::to_string(params->getFloat("V", params->getFloat("V_default", 400 * 6.68458712e-9))) + " au / s");
	spdlog::info("dt:" + std::to_string(params->getFloat("dt", params->getFloat("dt_default", 5.0f))) + " s");
}

std::string ParseParams::getTransformationTableName(std::string modelName)
{
	if ((modelName.compare("TwoDimensionBp") == 0) || (modelName.compare("ThreeDimensionBp") == 0))
	{
		return "SolarProp_K0_phi_table.csv";
	}
	else 
	{
		return "K0_phi_table.csv";
	}
	return NULL;
}