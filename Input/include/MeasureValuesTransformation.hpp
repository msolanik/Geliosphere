/**
 * @file MeasureValuesTransformation.hpp
 * @author Michal Solanik
 * @brief Extract measured parameters for simulation from table
 * @version 0.1
 * @date 2021-09-25
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef MEASURE_VALUES_TRANSFORMATION_H
#define MEASURE_VALUES_TRANSFORMATION_H

#include <string>
#include "ParamsCarrier.hpp"

/**
 * @brief MeasureValuesTransformation is responsible for calculating values of solar 
 * wind speed and diffusion coeficient for given month and year.
 * 
 */
class MeasureValuesTransformation
{
public:
    /**
	 * @brief Construct MeasureValuesTransformation object with path to table containing measured data for simulations. 
	 * 
	 * @param pathToTransformationTable Path to .csv file containing transformation table.
	 * @param model Type of the model
	 */
    MeasureValuesTransformation(std::string pathToTransformationTable, std::string model);

    /**
	 * @brief Calculate solar wind speed based on month and year. 
	 * 
	 * @param month Month of the year.
     * @param year Year ranging between 1951 - 2016
	 */
    float getSolarWindSpeedValue(int month, int year);

    /**
	 * @brief Calculate diffusion coefficient based on month and year. 
	 * 
	 * @param month Month of the year.
     * @param year Year ranging between 1951 - 2016
	 */
    float getDiffusionCoefficientValue(int month, int year);

	/**
	 * @brief Get tilt angle based on month and year. 
	 * 
	 * @param month Month of the year.
	 * @param year Year ranging between 1976 - 2015
	 */
	float getTiltAngle(int month, int year);

	/**
	 * @brief Get polarity based on month and year. 
	 * 
	 * @param month Month of the year.
	 * @param year Year ranging between 1976 - 2015
	 */
	int getPolarity(int month, int year);
private:
    /**
	 * @brief Path to file containing transformation table. 
	 * 
	 */
    std::string pathToTransformationTable;

	/**
	 * @brief Type of the model.
	 * 
	 */
	std::string model;

	/**
	 * @brief Get transformed value for given month.
	 * 
	 * @param month Month number
	 * @return Transformed value for given month.
	 */
	std::string getMonthName(int month);

	/**
	 * @brief Get the string identifier for csv table based on model.
	 * 
	 * @param month Month number
	 * @param year Year ranging between 1951 - 2016.
	 * @return String identifier for csv table based on model. 
	 */
	std::string getRowIdentifier(int month, int year);
	
	/**
	 * @brief Get the Carrington Rotation number in string.
	 * 
	 * @param month Month number
	 * @param year Year ranging between 1976 - 2015
	 * @return Carrington Rotation number in string 
	 */
	std::string getCarringtonRotation(int month, int year);
};

#endif