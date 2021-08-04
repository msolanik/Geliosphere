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