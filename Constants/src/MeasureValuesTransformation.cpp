#include "MeasureValuesTransformation.hpp"

MeasureValuesTransformation::MeasureValuesTransformation(std::string pathToUsoskinTable)
{
    this->pathToUsoskinTable = pathToUsoskinTable;
}

float MeasureValuesTransformation::getSolarWindSpeedValue(int month, int year)
{
    return 0.0f;
}

float MeasureValuesTransformation::getDiffusionCoefficientValue(int month, int year)
{
    return 0.0f;
}