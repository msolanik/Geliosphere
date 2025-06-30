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
	 * @brief Parse state enumeration for state machine pattern.
	 */
	enum class ParseState {
		SETUP,
		PARSING, 
		VALIDATION,
		PROCESSING,
		COMPLETE,
		ERROR
	};

	/**
	 * @brief Configuration structure to hold parsing parameters.
	 */
	struct ParseConfig {
		std::string inputFile;
		std::string pathToLogFile;
		float newDt, newK0, newV;
		int month, year;
		std::string newDestination, settings, customModelString;
		int numberOfTestParticles;
		std::string currentApplicationPath;
	};

	/**
	 * @brief Setup CLI options for the application.
	 * 
	 * @param app CLI::App instance to setup options on
	 * @param config Configuration structure containing all parsing parameters
	 */
	void setupCliOptions(CLI::App& app, ParseConfig& config);

	/**
	 * @brief Setup option relationships (excludes, requires).
	 */
	void setupOptionRelationships();

	/**
	 * @brief Execute state machine transition.
	 * 
	 * @param currentState Current parsing state
	 * @param config Configuration structure
	 * @param app CLI app instance
	 * @param argc Argument count
	 * @param argv Argument values
	 * @return Next state
	 */
	ParseState executeState(ParseState currentState, ParseConfig& config, CLI::App& app, int argc, char** argv);

	/**
	 * @brief Setup state handler.
	 * 
	 * @param config Configuration structure
	 * @param app CLI app instance
	 * @return Next state
	 */
	ParseState handleSetupState(ParseConfig& config, CLI::App& app);

	/**
	 * @brief Parsing state handler.
	 * 
	 * @param app CLI app instance
	 * @param argc Argument count
	 * @param argv Argument values
	 * @return Next state
	 */
	ParseState handleParsingState(CLI::App& app, int argc, char** argv);

	/**
	 * @brief Validation state handler.
	 * 
	 * @return Next state
	 */
	ParseState handleValidationState();

	/**
	 * @brief Processing state handler.
	 * 
	 * @param config Configuration structure
	 * @return Next state
	 */
	ParseState handleProcessingState(const ParseConfig& config);

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