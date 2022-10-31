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
	 */
    MeasureValuesTransformation(std::string pathToTransformationTable);

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
	 * @brief Calculate diffusion coefficient based on month and year. 
	 * 
	 * @param month Month of the year.
	 * @param year Year ranging between 1976 - 2015
	 */
	float getTiltAngle(int month, int year);
private:
    /**
	 * @brief Path to file containing transformation table. 
	 * 
	 */
    std::string pathToTransformationTable;

	/**
	 * @brief Get transformed value for given month.
	 * 
	 * @param month Month number
	 * @return Transformed value for given month.
	 */
	std::string getMonthName(int month);
};

#endif