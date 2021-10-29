/**
 * @file TomlSettings.hpp
 * @author Michal Solanik
 * @brief Parse values from settings 
 * @version 0.1
 * @date 2021-10-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef TOML_SETTINGS_H
#define TOML_SETTINGS_H

#include <string>
#include "ParamsCarrier.hpp"

/**
 * @brief TomlSettings is responsible for parsing settings from input file.
 * 
 */
class TomlSettings
{
public:
    /**
	 * @brief Construct TomlSettings object with path to file with settings. 
	 * 
	 * @param pathToSettingsFile Path to .toml file with settings.
	 */
    TomlSettings(std::string pathToSettingsFile);

    /**
	 * @brief Parse settings from settings file. 
	 * 
	 * @param paramsCarrier Object holding all input paramaters.
	 */
    void parseFromSettings(ParamsCarrier *paramsCarrier);
private:
    /**
	 * @brief Path to table with settings. 
	 * 
	 */
    std::string pathToSettingsFile;
};

#endif