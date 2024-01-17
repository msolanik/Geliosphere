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


using namespace std;

std::vector<std::string> splitString(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);
    std::string token;
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::vector<std::string>> readFile(ParamsCarrier *singleTone){
    std::ifstream file("../Algorithm/src/geliosphere_batch_input_paramaters.csv");

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return std::vector<std::vector<std::string>>();;
    }

    // Vector of vectors to store the loaded integers
    std::vector<std::vector<std::string>> data;
    
    // Read and process each line
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        // Split the line into fields
        std::vector<std::string> fields = splitString(line, ',');

        std::vector<std::string> row;
        // Process each field
        for (const std::string &field : fields) {
            row.push_back(field);
            // Do something with the field
            //std::cout << field << " ";
        }
        data.push_back(row);
        // Move to the next line
        //std::cout << std::endl;
    }

    // Close the file
    file.close();

    // Print the loaded 2D array
    /*std::cout << "Loaded 2D array:" << std::endl;
    for (const auto &row : data) {
        for (std::string value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    */
    return data;


}

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
    //std::cout << toml::toml_formatter{ tbl } << "\n";
    //std::cout << tbl << "\n";
    // Save to TOML file
    //std::remove("./Settings_batch.toml");
    //std::cout << "Snazim sa zapisat do Toml settings" << std::endl;
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
    //std::string currentApplicationPath = getApplicationPath(argv);
    //std::cout << "Hello, World! , value> " << singleTone->getString("inputFile", "test") << std::endl;
    //std::ifstream file(singleTone->getString("inputFile", "geliosphere_batch_input_paramaters.csv"));
    ParseParams *parse = new ParseParams();
    std::vector<std::vector<std::string>> data = readFile(singleTone);
	
    singleTone->putInt("csv", 1);
    singleTone->putString("model", "2D Geliosphere");
    parse->dtSetCheck(singleTone,1000);
    //ParamsCarrier* single = singleTone->instance();
    std::cout << "singletone inner things: " << singleTone->getString("model", "daco nedobre") <<std::endl;
    std::size_t numRows = data.size();
    std::cout << "Number of rows: " << numRows << std::endl;
    std::string  targetDirectory= "test";    
    int test = 0;

    for (int i = 0; i < numRows; i++) {
        if(std::stod(data[i][0]) >= 1997 && std::stod(data[i][0]) <= 1998){
            generateTomlFile(std::stod(data[i][3]),std::stod(data[i][4]));
            parse->newSettingsLocationCheck(singleTone, "./Settings_batch.toml");
            //std::cout << std::filesystem::absolute("./") << std::endl;
            parse->monthYearCheck(singleTone, std::stod(data[i][0]), std::stod(data[i][1]), "./" );
            targetDirectory = data[i][0] + "_" + data[i][1] + "_" + data[i][2];
            singleTone->putString("destination", targetDirectory);
            std::cout << "Destination: " << singleTone->getString("destination", "test") << " : " << data[i][0] << "_" << data[i][1] << "_" << data[i][2] << std::endl;
            std::cout << "Model: " << singleTone->getString("model", "test") << std::endl;
            AbstractAlgorithm *actualAlgorithm = new GeliosphereAlgorithm();
            actualAlgorithm->runAlgorithm(singleTone);

            for (int j = 0; j < 5; j++){    
                std::cout << " " << data[i][j];
            }
            std::cout << std::endl;
            test++;
            chdir("../../..");
        }
    }
    std::cout << "Test: " << test << std::endl;
}



