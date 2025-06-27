/**
 * @file ParseParams.hpp
 * @author Michal Solanik
 * @brief Parser of arguments from CLI
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef PARSE_PARAMS_H
#define PARSE_PARAMS_H

#include <string>
#include "ParamsCarrier.hpp"
#include "CLI/App.hpp"
#include "CLI/Option.hpp"

/**
 * @brief ParseParams is responsible for parsing arguments from CLI.
 * 
 */
class ParseParams
{
public:
	/**
	 * @brief Parse params from CLI.
	 * 
	 * @param argc Number of arguments
	 * @param argv Arguments
	 * @return 1 in the case of successfully parsed arguments and
	 * -1 in the case of failure. 
	 */
	int parseParams(int argc, char **argv);

	/**
	 * @brief Get the Params object
	 * 
	 * @return ParamsCarrier* with parsed arguments
	 */
	ParamsCarrier *getParams();
	
private:
	/**
	 * @brief Instance of ParamsCarrier for placing 
	 * parsed arguments.
	 * 
	 */
	ParamsCarrier *singleTone;

	/**
	 * @brief Print basic parameters which are used in simulation.
	 * 
	 * @param params Data structure holding input data.
	 */
	void printParameters(ParamsCarrier *params);

	/**
	 * @brief Return path where Geliosphere is located.
	 * 
	 * @param argv Geliosphere parameters
	 * @return Path where Geliosphere is located
	 */
	std::string getApplicationPath(char **argv);

	/**
	 * @brief Setup CLI options for the application.
	 * 
	 * @param app CLI::App instance to setup options on
	 * @param inputFile Reference to input file string
	 * @param pathToLogFile Reference to log file path string
	 * @param newDt Reference to new dt value
	 * @param newK0 Reference to new K0 value
	 * @param newV Reference to new V value
	 * @param month Reference to month value
	 * @param year Reference to year value
	 * @param newDestination Reference to destination string
	 * @param settings Reference to settings string
	 * @param customModelString Reference to custom model string
	 * @param numberOfTestParticles Reference to number of test particles
	 */
	void setupCliOptions(CLI::App& app, std::string& inputFile, std::string& pathToLogFile,
		float& newDt, float& newK0, float& newV, int& month, int& year,
		std::string& newDestination, std::string& settings, std::string& customModelString,
		int& numberOfTestParticles);

	/**
	 * @brief Setup option relationships (excludes, requires).
	 */
	void setupOptionRelationships();

	/**
	 * @brief Process general options like csv, run_simulation, destination.
	 * 
	 * @param pathToLogFile Log file path
	 * @param newDestination Destination path
	 * @return 1 on success, -1 on failure
	 */
	int processGeneralOptions(const std::string& pathToLogFile, const std::string& newDestination);

	/**
	 * @brief Process value options like dt, K0, V, numberOfTestParticles.
	 * 
	 * @param newDt New dt value
	 * @param newK0 New K0 value
	 * @param newV New V value
	 * @param numberOfTestParticles Number of test particles
	 * @return 1 on success, -1 on failure
	 */
	int processValueOptions(float newDt, float newK0, float newV, int numberOfTestParticles);

	/**
	 * @brief Process model selection options.
	 * 
	 * @param customModelString Custom model string
	 * @param inputFile Input file for batch run
	 * @return 1 on success, -1 on failure
	 */
	int processModelOptions(const std::string& customModelString, const std::string& inputFile);

	/**
	 * @brief Process settings and time-related options.
	 * 
	 * @param settings Settings file path
	 * @param month Month value
	 * @param year Year value
	 * @param currentApplicationPath Current application path
	 * @return 1 on success, -1 on failure
	 */
	int processSettingsOptions(const std::string& settings, int month, int year, 
		const std::string& currentApplicationPath);

	// CLI option pointers - stored as class members to be accessible across functions
	CLI::Option *forwardModel;
	CLI::Option *backwardModel;
	CLI::Option *solarPropLikeModel;
	CLI::Option *geliosphereModel;
	CLI::Option *csv;
	CLI::Option *run_simulation;
	CLI::Option *cpuOnly;
	CLI::Option *batchRun;
	CLI::Option *dtset;
	CLI::Option *kset;
	CLI::Option *vset;
	CLI::Option *destination;
	CLI::Option *setNumberOfTestParticles;
	CLI::Option *monthOption;
	CLI::Option *yearOption;
	CLI::Option *settingsOption;
	CLI::Option *customModel;
};

#endif