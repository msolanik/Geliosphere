/**
 * @file InputValidation.hpp
 * @author Tomas Telepcak, Michal Solanik
 * @brief Utilities for validating input data, creating input settings
 * @version 1.2.0
 * @date 2024-01-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef INPUT_VALIDATION
#define INPUT_VALIDATION

#include "ParamsCarrier.hpp"

/**
 * @brief Class responsible for validating and setting input values for simulations.
 * 
 */
class InputValidation
{
public:
	/**
	 * @brief Parse values from TOML settings file.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param settings Path to settings file. 
	 */
	void newSettingsLocationCheck(ParamsCarrier *singleTone, std::string settings);

	/**
	 * @brief Set the time step dt.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param dt time step dt value.
	 */
	void setDt(ParamsCarrier *singleTone, float dt);

	/**
	 * @brief Validate value of time step dt.
	 * 
	 * @param dt Value of time step dt.
	 * @return true if value is valid.
	 * @return false if value is not valid.
	 */
	bool checkDt(float dt);

	/**
	 * @brief Set the diffusion coefficient K0.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param K0 diffusion coefficient K0.
	 */
	void setK0(ParamsCarrier *singleTone, float K0);

	/**
	 * @brief Validate value of diffusion coefficient K0.
	 * 
	 * @param K0 Value of diffusion coefficient K0.
	 * @return true if value is valid.
	 * @return false if value is not valid.
	 */
	bool checkK0(float K0);

	/**
	 * @brief Set the solar wind speed V.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param V solar wind speed V.
	 */
	void setV(ParamsCarrier *singleTone, float V);

	/**
	 * @brief Validate value of solar wind speed V.
	 * 
	 * @param V Value of solar wind speed V.
	 * @return true if value is valid.
	 * @return false if value is not valid.
	 */
	bool checkV(float V);

	/**
	 * @brief Set the Number Of Test Particles in millions.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param numberOfTestParticles number of test particles in millions.
	 */
	void setNumberOfTestParticles(ParamsCarrier *singleTone, int numberOfTestParticles);

	/**
	 * @brief Validate value of number of test particles.
	 * 
	 * @param numberOfTestParticles value of number of test particles in millions.
	 * @return true if value is valid.
	 * @return false if value is not valid.
	 */
	bool checkNumberOfTestParticles(int numberOfTestParticles);

	/**
	 * @brief Check and set K0, V (1D models only), tilt angle (2D models only), 
	 * and polarity (2D models only) depending on the model.
	 * 
	 * @param singleTone datastructure containing input parameters.
	 * @param year selected year.
	 * @param month selected month.
	 * @param currentApplicationPath current path on which is application executed.
	 */
	void monthYearCheck(ParamsCarrier *singleTone, int year, int month,std::string currentApplicationPath);

	/**
	 * @brief Get name of transformation table for selected model.
	 * 
	 * @param modelName name of the model.
	 * @return name of transformation table. 
	 */
	std::string getTransformationTableName(std::string modelName);

	/**
	 * @brief Check if input model is SolarProp-like model.
	 * 
	 * @param modelName name of the model.
	 * @return true if input model is SolarProp-like model.
	 * @return false if input model is not SolarProp-like model.
	 */
	bool isInputSolarPropLikeModel(std::string modelName);

	/**
	 * @brief Check if input model is Geliosphere 2D model.
	 * 
	 * @param modelName name of the model.
	 * @return true if input model is Geliosphere 2D model.
	 * @return false if input model is not Geliosphere 2D model.
	 */
	bool isInputGeliosphere2DModel(std::string modelName);
};


#endif 