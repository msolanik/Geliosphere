#include "MeasureValuesTransformation.hpp"
#include "rapidcsv.h"

MeasureValuesTransformation::MeasureValuesTransformation(std::string pathToTransformationTable)
{
    this->pathToTransformationTable = pathToTransformationTable;
}

float MeasureValuesTransformation::getSolarWindSpeedValue(int month, int year)
{
    rapidcsv::Document doc(pathToTransformationTable, rapidcsv::LabelParams(0, 0));
    return doc.GetCell<float>("V", std::to_string(year).append(".").append(getMonthName(month)));
}

float MeasureValuesTransformation::getDiffusionCoefficientValue(int month, int year)
{
    rapidcsv::Document doc(pathToTransformationTable, rapidcsv::LabelParams(0, 0));
    return doc.GetCell<float>("k0_au2/s", std::to_string(year).append(".").append(getMonthName(month)));
}

std::string MeasureValuesTransformation::getMonthName(int month) 
{
    switch (month)
    {
        case 1:
            return "04";
        case 2:
            return "12";
        case 3:
            return "20";
        case 4:
            return "29";
        case 5:
            return "37";
        case 6:
            return "45";
        case 7:
            return "54";
        case 8:
            return "62";
        case 9:
            return "70";
        case 10:
            return "79";
        case 11:
            return "87";
        case 12:
            return "95";
        default:
            return "NaN";
    }
}