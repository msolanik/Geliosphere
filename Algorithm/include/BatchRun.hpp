/**
 * @file BatchRun.hpp
 * @author Tomas Telepcak, Michal Solanik
 * @brief Abstract definition for algorithm
 * @version 1.2.0
 * @date 2024-01-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BATCH_RUN
#define BATCH_RUN

#include "AbstractAlgorithm.hpp"

#include <limits>

#include "rapidcsv.h"
#include "toml.hpp"

#include "InputValidation.hpp"

/**
 * @brief Data structure representing input into batch processing. 
 * All fields should be present in input CSV file.
 * 
 */
struct batchRunInput
{
	int year;
	int month;
	double r;
	double theta;
	double V;
	double K0;
	double dt;
	long N;
	std::string model;
	std::string pathToCustomSettingsFile;
	std::string name;
};

/**
 * @brief Class implements @ref AbstractAlgorithm "AbstractAlgorithm" interface 
 * to define support functions for running batch mode.
 * 
 */
class BatchRun : public AbstractAlgorithm
{
public:
	/**
	 * @brief Run batch mode.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 */
	void runAlgorithm(ParamsCarrier *singleTone);
	
	/**
	 * @brief Construct a new Batch Run object, initializing input validation.
	 * 
	 */
	BatchRun();
private:
	/**
	 * @brief Instance of class used for validating input. 
	 * 
	 */
	InputValidation *inputValidation;

	/**
	 * @brief Generate new TOML file in tmp location.
	 * 
	 * @param r Input r injection value.
	 * @param theta Input theta injection value.
	 * @param pathToSettingsFile Path to settings file.
	 */
	void generateTomlFile(double r, double theta, std::string pathToSettingsFile);

	/**
	 * @brief Get the default file settings, if there is no file present, function returns 
	 * default settings values.
	 * 
	 * @param pathToSettingsFile Path to TOML settings file. 
	 * @return Object containing all fields contained in TOML settings file.
	 */
	toml::value getDefaultSettingsFile(std::string pathToSettingsFile);

	/**
	 * @brief Retrieve values for single simulation from input CSV file.
	 * 
	 * @param batchRunInput Input for single simulation, input from single line from CSV are stored 
	 * 	in this data structure.
	 * @param doc Input for batch mode in form of CSV file. 
	 * @param row Index of currently processed row.
	 */
	void getValuesForSingleSimulation(struct batchRunInput* batchRunInput, rapidcsv::Document doc, int row);

	/**
	 * @brief Validate values in columns in input batch file.
	 * 
	 * @param doc Input for batch mode in form of CSV file. 
	 * @return true if batch file is valid.
	 * @return false if batch file file is not valid.
	 */
	bool validateInputBatchFile(rapidcsv::Document doc);

	/**
	 * @brief Validate input values for simulation.
	 * 
	 * @param batchRunInput Input for single simulation from batch file.
	 * @return true if input for single simulation is valid.
	 * @return false if input for single simulation is not valid.
	 */
	bool validateInputForSimulation(struct batchRunInput* batchRunInput);

	/**
	 * @brief Parse values from @ref batchRunInput "batchRunInput" structure into @ref ParamsCarrier "ParamsCarrier".
	 * 
	 * @param singleTone Datastructure containing input parameters.
	 * @param batchRunInput Input for single simulation from batch file.
	 * @return true if values were parsed correctly.
	 * @return false if values weren not parsed correctly.
	 */
	bool parseValues(ParamsCarrier *singleTone, struct batchRunInput* batchRunInput);

	/**
	 * @brief Retrieve instance of supported model for execution. Some models may require specific input, which 
	 * may require additional changes to batch mode input.
	 * 
	 * @param name Name of the model.
	 * @return AbstractAlgorithm* Retrieve instance of supported model for execution.
	 */
	AbstractAlgorithm* getSupportedModel(std::string name);
};

#endif