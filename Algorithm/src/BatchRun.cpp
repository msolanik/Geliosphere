#include "BatchRun.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "spdlog/spdlog.h"

#include "GeliosphereAlgorithm.hpp"
#include "OneDimensionFpAlgorithm.hpp"
#include "OneDimensionBpAlgorithm.hpp"
#include "ParamsCarrier.hpp"
#include "ParseParams.hpp"
#include "SolarPropLikeAlgorithm.hpp"

BatchRun::BatchRun()
{
    inputValidation = new InputValidation();
}

void BatchRun::runAlgorithm(ParamsCarrier *singleTone)
{
    std::set<std::string> excludedItems({"default_settings_file", "currentApplicationPath", "csv", "isCpu", "inputBatchFile"});

    std::string pathToBatchFile = singleTone->getString("inputBatchFile", "../geliosphere_batch_input_paramaters.csv");
    std::ifstream batchFile(pathToBatchFile.c_str());

    if(!batchFile.good())
    {
        spdlog::error("Cannot open input batch file.");
        return;
    }

    rapidcsv::Document doc(pathToBatchFile, rapidcsv::LabelParams(0, -1), rapidcsv::SeparatorParams(), rapidcsv::ConverterParams(true, -1.0, -1));    
    
    AbstractAlgorithm *actualAlgorithm;

    if (!validateInputBatchFile(doc))
    {
        spdlog::error("Input batch file contains unrecoverable errors.");
        return;
    }

    singleTone->eraseAllItems(excludedItems);
    for (int i = 0; i < doc.GetRowCount() ; i++) {
        struct batchRunInput input;
        getValuesForSingleSimulation(&input, doc, i);
        if (!validateInputForSimulation(&input))
        {
            spdlog::error("Skipping incorrect {}. simulation.", i + 1);
            continue;
        }
        if (!parseValues(singleTone, &input))
        {
            spdlog::error("Unable to parse values for {}. simulation.", i + 1);
            continue;    
        }
        
        singleTone->putString("model", input.model);
        actualAlgorithm = getSupportedModel(input.model);
        if (actualAlgorithm == NULL)
        {
            spdlog::error("Selected model for simulation {} is not supported in batch mode.", i + 1);
            continue;
        }
        actualAlgorithm->runAlgorithm(singleTone);
        if (chdir("../../..") == -1)
        {
            spdlog::error("Could not change directory");
            break;
        } 
        singleTone->eraseAllItems(excludedItems);
    }
}

AbstractAlgorithm* BatchRun::getSupportedModel(std::string name)
{
    if (name.compare("1D Fp") == 0)
	{
		return new OneDimensionFpAlgorithm();
	}
	else if (name.compare("1D Bp") == 0)
	{
		return new OneDimensionBpAlgorithm();
	}
	else if (name.compare("2D SolarProp-like") == 0)
	{
		return new SolarPropLikeAlgorithm();
	}
	else if (name.compare("2D Geliosphere") == 0)
	{
		return new GeliosphereAlgorithm();
	}
	return NULL;
}
	
void BatchRun::getValuesForSingleSimulation(struct batchRunInput* batchRunInput, rapidcsv::Document doc, int row)
{
    auto columnNames = doc.GetColumnNames();
    if (std::find(columnNames.begin(), columnNames.end(), "name") != columnNames.end())
    {
        batchRunInput->name = doc.GetCell<std::string>("name", row);
    }
    else
    {
        spdlog::warn("Column name is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "month") != columnNames.end())
    {
        batchRunInput->month = doc.GetCell<int>("month", row);
    }
    else
    {
        spdlog::warn("Column month is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "year") != columnNames.end())
    {
        batchRunInput->year = doc.GetCell<int>("year", row);
    }
    else
    {
        spdlog::warn("Column year is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "V") != columnNames.end())
    {
        batchRunInput->V = doc.GetCell<float>("V", row);
    }
    else
    {
        spdlog::warn("Column V is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "K0") != columnNames.end())
    {
        batchRunInput->K0 = doc.GetCell<float>("K0", row);
    }
    else
    {
        spdlog::warn("Column K0 is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "dt") != columnNames.end())
    {
        batchRunInput->dt = doc.GetCell<float>("dt", row);
    }
    else
    {
        spdlog::warn("Column dt is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "r") != columnNames.end())
    {
        batchRunInput->r = doc.GetCell<float>("r", row);
    }
    else
    {
        spdlog::warn("Column r is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "theta") != columnNames.end())
    {
        batchRunInput->theta = doc.GetCell<float>("theta", row);
    }
    else
    {
        spdlog::warn("Column theta is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "N") != columnNames.end())
    {
        batchRunInput->N = doc.GetCell<long>("N", row);
    }
    else
    {
        spdlog::warn("Column N is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "model") != columnNames.end())
    {
        batchRunInput->model = doc.GetCell<std::string>("model", row);
    }
    else
    {
        spdlog::warn("Column model is not defined.");
    }
    if (std::find(columnNames.begin(), columnNames.end(), "pathToCustomSettingsFile") != columnNames.end())
    {
        batchRunInput->pathToCustomSettingsFile = doc.GetCell<std::string>("pathToCustomSettingsFile", row);
    }
    else
    {
        spdlog::warn("Column pathToCustomSettingsFile is not defined.");
    }
}

void BatchRun::generateTomlFile(double r, double theta, std::string pathToSettingsFile) {
    toml::value tbl = getDefaultSettingsFile(pathToSettingsFile);

    if (r > 0.0)
    {
        tbl["default_values"]["r_injection"] = r;
    }

    if (theta > 0.0)
    {
        tbl["2d_models_common_settings"]["theta_injection"] = theta;
    }

    std::ofstream file("/tmp/Settings_batch.toml");

    if (file.is_open()) {
        file << tbl;
        file.close();
    } else {
        spdlog::error("Could not open Settings file for writing");
    }
}

toml::value BatchRun::getDefaultSettingsFile(std::string pathToSettingsFile)
{
    std::ifstream f(pathToSettingsFile.c_str());

    if(f.good())
    {
        return toml::parse(pathToSettingsFile);
    }

    return toml::table{
        { "default_values", toml::table{
            { "uniform_energy_injection_maximum", 101.0 } ,
            { "K0", 5e+22 },
            { "V", 400.0 },
            { "dt", 50.0 },
            { "r_injection", 1.0 }} 
        },
        { "2d_models_common_settings", toml::table{
            { "theta_injection", 90.0 } ,
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
}

bool BatchRun::validateInputForSimulation(struct batchRunInput* batchRunInput)
{
    bool is2Dmodel = inputValidation->isInputGeliosphere2DModel(batchRunInput->model)
     || inputValidation->isInputSolarPropLikeModel(batchRunInput->model);
    if (!is2Dmodel && batchRunInput->theta != -1.0) 
    {
        spdlog::warn("Setting of theta for 1D models does not have any effect on simulation");
    }
    if ((batchRunInput->month != -1.0 && batchRunInput->year == -1.0) || 
        (batchRunInput->month == -1.0 && batchRunInput->year != -1.0))
    {
        spdlog::error("Both month and year must be set.");
        return false;
    }
    bool isSimulationDateSet = (batchRunInput->month != -1.0 && batchRunInput->year != -1.0);
    if (isSimulationDateSet && (batchRunInput->K0 != -1.0 || batchRunInput->V != -1.0))
    {
        spdlog::error("Simulation dates cannot be combined with K0 or V.");
        return false;
    }
    if (batchRunInput->model.compare("batch run") == 0)
    {
        spdlog::error("Cannot select batch run as model.");
        return false;
    }
    return true;
}

bool BatchRun::parseValues(ParamsCarrier *singleTone, struct batchRunInput* batchRunInput)
{
    if (batchRunInput->dt != -1.0)
	{
		if (!inputValidation->checkDt(batchRunInput->dt))
		{
			spdlog::error("dt is out of range!(3-5000)");
			return false;
		}
		inputValidation->setDt(singleTone, batchRunInput->dt);
	}
	if (batchRunInput->K0 != -1.0)
	{
		if (!inputValidation->checkK0(batchRunInput->K0))
		{
			spdlog::error("K0 is out of range!(>0)");
			return false;
		}
		if (batchRunInput->K0 < 1e19 || batchRunInput->K0 > 1e23)
		{
			spdlog::warn("K0 is out of recommended range!(1e19-1e23 cm^2/s)");
		}
		inputValidation->setK0(singleTone, batchRunInput->K0);
	}
	if (batchRunInput->N != -1)
	{
		if (!inputValidation->checkNumberOfTestParticles(batchRunInput->N))
		{
			spdlog::error("Number of test particles must be greater than 0!");
			return false;
		}
		inputValidation->setNumberOfTestParticles(singleTone, batchRunInput->N);
	}
	if (batchRunInput->V != -1.0)
	{
		if (!inputValidation->checkV(batchRunInput->V))
		{
			spdlog::error("V is out of range!(100-1500 km/s)");
			return -1;
		}
		inputValidation->setV(singleTone, batchRunInput->V);
	}
    if (!batchRunInput->name.empty())
    {
        if (batchRunInput->name.find_first_not_of(' ') != std::string::npos)
        {
            singleTone->putString("destination", batchRunInput->name);
        }				
        else
        {
			spdlog::warn("Name containing whitespaces is not valid, using auto generated name instead.");           
        }
    }
    generateTomlFile(batchRunInput->r, batchRunInput->theta, batchRunInput->pathToCustomSettingsFile);
    inputValidation->newSettingsLocationCheck(singleTone, "/tmp/Settings_batch.toml");
    if (batchRunInput->month != -1 && batchRunInput->year != -1)
    {
        inputValidation->monthYearCheck(singleTone, batchRunInput->year, batchRunInput->month, singleTone->getString("currentApplicationPath", "./"));
    }
    return true;
}

bool BatchRun::validateInputBatchFile(rapidcsv::Document doc)
{
    // Check uniqueness of names in name column
    std::vector<std::string> col = doc.GetColumn<std::string>("name");
    col.erase(
        std::remove_if(
            col.begin(),
            col.end(),
            [](std::string const& s) 
            { 
                return s.empty() || s.find_first_not_of(' ') == std::string::npos;
            }),
        col.end());
    std::sort(col.begin(), col.end());
    if (std::adjacent_find(col.begin(), col.end()) != col.end())
    {
        spdlog::error("Input batch file contains duplicated values. Use unique names for simulations, or leave fields empty for using autogenerated values.");
        return false;
    }

    return true;
}