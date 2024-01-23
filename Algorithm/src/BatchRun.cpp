#include "BatchRun.hpp"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <toml++/toml.hpp>
#include <toml++/toml.h>
#include "ParseParams.hpp"
#include "ParamsCarrier.hpp"
#include <cstdlib>
#include <filesystem>
#include "GeliosphereAlgorithm.hpp"
#include "rapidcsv.h"
#include "InputValidation.hpp"
#include "AbstractAlgorithmFactory.hpp"
#include "spdlog/spdlog.h"

using namespace std;

void generateTomlFile(double r, double theta) {
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
    
}

void BatchRun::runAlgorithm(ParamsCarrier *singleTone)
{
    InputValidation *inputValidation = new InputValidation();
    rapidcsv::Document doc("../geliosphere_batch_input_paramaters.csv", rapidcsv::LabelParams(0, -1));    
     
    singleTone->putInt("csv", 1);
    inputValidation->dtSetCheck(singleTone, 1000);
    
    std::string  targetDirectory= "test";    
    AbstractAlgorithm *actualAlgorithm;
    AbstractAlgorithmFactory *factory = AbstractAlgorithmFactory::CreateFactory(AbstractAlgorithmFactory::TYPE_ALGORITHM::COSMIC);

    for (int i = 0; i < doc.GetRowCount() ; i++) {
        std::vector<std::string> data = doc.GetRow<std::string>(i);
        
        if(std::stod(data[0]) >= 1997 && std::stod(data[0]) <= 1998){
            generateTomlFile(std::stod(data[3]),std::stod(data[4]));
            singleTone->putString("model", data[5]);
            inputValidation->newSettingsLocationCheck(singleTone, "./Settings_batch.toml");
            inputValidation->monthYearCheck(singleTone, std::stod(data[0]), std::stod(data[1]), "./" );
            targetDirectory = data[0] + "_" + data[1] + "_" + data[2];
            singleTone->putString("destination", targetDirectory);
            
            actualAlgorithm = factory->getAlgorithm(singleTone->getString("model", "1D Fp"));
            if (actualAlgorithm == NULL)
            {
                spdlog::error("Selected custom model is not defined in factory.");
                break;
            }
            actualAlgorithm->runAlgorithm(singleTone);
            chdir("../../..");
        }
    }
}



