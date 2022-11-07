#include "MeasureValuesTransformation.hpp"

#include "rapidcsv.h"

#include <ctime>
#include <cmath>

MeasureValuesTransformation::MeasureValuesTransformation(std::string pathToTransformationTable, std::string model)
{
    this->pathToTransformationTable = pathToTransformationTable;
    this->model = model;
}

float MeasureValuesTransformation::getSolarWindSpeedValue(int month, int year)
{
    rapidcsv::Document doc(pathToTransformationTable, rapidcsv::LabelParams(0, 0));
    return doc.GetCell<float>("V", getRowIdentifier(month, year));
}

float MeasureValuesTransformation::getDiffusionCoefficientValue(int month, int year)
{
    rapidcsv::Document doc(pathToTransformationTable, rapidcsv::LabelParams(0, 0));
    return doc.GetCell<float>("k0_au2/s", getRowIdentifier(month, year));
}

float MeasureValuesTransformation::getTiltAngle(int month, int year)
{
    rapidcsv::Document doc(pathToTransformationTable, rapidcsv::LabelParams(0, 0));
    return doc.GetCell<float>("tilt_angle", getRowIdentifier(month, year));
}

std::string MeasureValuesTransformation::getRowIdentifier(int month, int year)
{
    if ((model.compare("TwoDimensionBp") == 0) || (model.compare("ThreeDimensionBp") == 0))
    {
        return getCarringtonRotation(month, year);
    }
    else
    {
        return std::to_string(year).append(".").append(getMonthName(month));
    }
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

std::string MeasureValuesTransformation::getCarringtonRotation(int month, int year)
{
    struct tm start;
    start.tm_year = 1976 - 1900;
    start.tm_mon = 5 - 1;
    start.tm_mday = 27;
    start.tm_hour = 0;
    start.tm_min = 0;
    start.tm_sec = 0;

    struct tm end;
    end.tm_year = year - 1900;
    end.tm_mon = month - 1;
    end.tm_mday = 15;
    end.tm_hour = 0;
    end.tm_min = 0;
    end.tm_sec = 0;

    double seconds = difftime(mktime(&end), mktime(&start));
    double diffInCarringtonRotations = seconds / 2356560.0;
    double roundedPartInCarringtonRotations; 
    double decimalPartInCarringtonRotations = std::modf(diffInCarringtonRotations, &roundedPartInCarringtonRotations);

    if (decimalPartInCarringtonRotations <= 0.5499)
    {
        diffInCarringtonRotations = std::trunc(diffInCarringtonRotations);
    }
    else 
    {
        diffInCarringtonRotations = std::ceil(diffInCarringtonRotations);
    }

    return std::to_string((int) diffInCarringtonRotations + 1642);
}