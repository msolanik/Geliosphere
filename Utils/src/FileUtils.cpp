#include <stdio.h>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include "FileUtils.hpp"
#include "ParamsCarrier.hpp"

int mkdirAndchdir(std::string directoryName)
{
	char *name = new char[directoryName.length() + 1];
	strcpy(name, directoryName.c_str());
	DIR *dir = opendir(name);
	if (dir)
	{
		int resultMkdir = chdir(name);
		if (resultMkdir != 0)
		{
			printf("ERROR %d: unable to change directory; %s\n", resultMkdir, strerror(resultMkdir));
			delete[] name;
			return -1;
		}
	}
	else if (ENOENT == errno)
	{
		int resultMkdir = mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (resultMkdir != 0)
		{
			printf("ERROR %d: unable to mkdir; %s\n", resultMkdir, strerror(resultMkdir));
			delete[] name;
			return -1;
		}
		resultMkdir = chdir(name);
		if (resultMkdir != 0)
		{
			printf("ERROR %d: unable to change directory; %s\n", resultMkdir, strerror(resultMkdir));
			delete[] name;
			return -1;
		}
	}
	else
	{
		printf("ERROR: unable to open directory");
		delete[] name;
		return -1;
	}
	delete[] name;
	return 1;
}

bool createDirectory(std::string methodDirectory, std::string destination)
{
	mode_t process_mask = umask(0);
	umask(process_mask);
	if (mkdirAndchdir("results") != 1)
	{
		return false;
	}
	if (mkdirAndchdir(methodDirectory) != 1)
	{
		return false;
	}
	if (mkdirAndchdir(destination) != 1)
	{
		return false;
	}
	return true;
}

std::string getDirectoryName(ParamsCarrier *singleTone)
{
	time_t clk = time(NULL);
	char *dirName = ctime(&clk);
	for (int i = 0; i < strlen(dirName); i++)
	{
		if (dirName[i] == ' ')
		{
			dirName[i] = '_';
		}
		else if (i == strlen(dirName) - 1)
		{
			dirName[i] = 0;
		}
	}
	std::string parameters;
	parameters += dirName;
	parameters += "_dt=" + std::to_string(singleTone->getFloat("dt", 5.0f)) + "K0=" +
				  singleTone->getString("K0input", "0.000222") + "V=" + singleTone->getString("Vinput", "400");
	return parameters;
}

void writeSimulationReportFile(ParamsCarrier *singleTone)
{
	FILE *file = fopen("simulation_log.txt", "w");
	fprintf(file, "Model: %s\n", singleTone->getString("model", "FWMethod").c_str());
	fprintf(file, "---------------------\n");
	if(singleTone->getInt("month_option", -1) != -1) 
	{
		fprintf(file, "Selected month: %d\n", singleTone->getInt("month_option", -1));
		fprintf(file, "Selected year: %d\n", singleTone->getInt("year_option", -1));
	}

	if (singleTone->getString("model", "FWMethod").compare("TwoDimensionBp") == 0)
	{
		fprintf(file, "SolarProp-like model ratio: %g\n", singleTone->getFloat("solarPropRatio", 0.02f));
	}
	if (singleTone->getString("model", "FWMethod").compare("ThreeDimensionBp") == 0)
	{
		fprintf(file, "Geliosphere model ratio: %g\n", singleTone->getFloat("geliosphereRatio", 0.2f));
		fprintf(file, "C Delta: %g\n", singleTone->getFloat("C_delta", 8.7e-5f));
		fprintf(file, "Loaded tilt angle from file: %g\n", singleTone->getFloat("tilt_angle", singleTone->getFloat("default_tilt_angle", 0.1)));
		fprintf(file, "Calculated tilt angle:: %g\n", singleTone->getFloat("tilt_angle", singleTone->getFloat("default_tilt_angle", 0.1)) * 3.1415926535f / 180.0f);
	}
	fprintf(file, "dt: %g\n", singleTone->getFloat("dt", singleTone->getFloat("dt_default", 50.0f)));
	fprintf(file, "K0: %g\n", singleTone->getFloat("K0", singleTone->getFloat("K0_default", 0.000222f)));
	fprintf(file, "V: %g\n", singleTone->getFloat("V", singleTone->getFloat("V_default", 2.66667e-6)));
	fclose(file);
}