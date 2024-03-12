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

void BatchRun::runAlgorithm(ParamsCarrier *singleTone)
{
    InputValidation *inputValidation = new InputValidation();
    rapidcsv::Document doc("../geliosphere_batch_input_paramaters.csv", rapidcsv::LabelParams(0, -1));    
    
    AbstractAlgorithm *actualAlgorithm;
    AbstractAlgorithmFactory *factory = AbstractAlgorithmFactory::CreateFactory(AbstractAlgorithmFactory::TYPE_ALGORITHM::COSMIC);

    for (int i = 0; i < doc.GetRowCount() ; i++) {
        std::vector<std::string> data = doc.GetRow<std::string>(i);
        
        if(std::stod(data[0]) >= 1997 && std::stod(data[0]) <= 1998){
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



