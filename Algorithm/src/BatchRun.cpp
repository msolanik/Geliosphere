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


using namespace std;

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
    
}

void BatchRun::runAlgorithm(ParamsCarrier *singleTone)
{
    //rapidcsv::Document doc("../Algorithm/src/geliosphere_batch_input_paramaters.csv", rapidcsv::LabelParams(0, -1));
    rapidcsv::Document doc("../geliosphere_batch_input_paramaters.csv", rapidcsv::LabelParams(0, -1));
    
    std::cout << "velkost nacitaneho suboru" << doc.GetRowCount() <<std::endl;
    std::cout << "pocet stlpcov" << doc.GetColumnCount() <<std::endl;
    
    std::vector<std::string> row = doc.GetRow<std::string>(125);
    std::cout << "test riadkov : " << row[4] << std::endl;      
    
    ParseParams *parse = new ParseParams();
    
    singleTone->putInt("csv", 1);
    singleTone->putString("model", "2D Geliosphere");
    parse->dtSetCheck(singleTone,1000);
    std::cout << "singletone inner things: " << singleTone->getString("model", "daco nedobre") <<std::endl;
    std::string  targetDirectory= "test";    
    int test = 0;
    int opakovanie = 0;

    for (int i = 0; i < doc.GetRowCount() ; i++) {
        std::vector<std::string> data = doc.GetRow<std::string>(i);
        if(std::stod(data[0]) >= 1997 && std::stod(data[0]) <= 1998){
            generateTomlFile(std::stod(data[3]),std::stod(data[4]));
            parse->newSettingsLocationCheck(singleTone, "./Settings_batch.toml");
            //std::cout << std::filesystem::absolute("./") << std::endl;
            parse->monthYearCheck(singleTone, std::stod(data[0]), std::stod(data[1]), "./" );
            targetDirectory = data[0] + "_" + data[1] + "_" + data[2];
            singleTone->putString("destination", targetDirectory);
            std::cout << "Destination: " << singleTone->getString("destination", "test") << " : " << data[0] << "_" << data[1] << "_" << data[2] << std::endl;
            std::cout << "Model: " << singleTone->getString("model", "test") << std::endl;
            AbstractAlgorithm *actualAlgorithm = new GeliosphereAlgorithm();
            actualAlgorithm->runAlgorithm(singleTone);

            for (int j = 0; j < 5; j++){    
                std::cout << " " << data[j];
            }
            std::cout << std::endl;
            test++;
            chdir("../../..");
        }
        opakovanie = i ;
    }
    std::cout << "Test: " << test << std::endl;
    std::cout << "I : " << opakovanie << std::endl;
}



