/**
 * @file main.cpp
 * @author Michal Solanik
 * @brief Main function
 * @version 0.1
 * @date 2021-07-09
 * 
 * @details Main function is used to parse arguments and
 * 	start new simulation with given parameters.  
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <stdlib.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <sys/types.h>

#include "ParamsCarrier.hpp"
#include "ParseParams.hpp"
#include "AbstractAlgorithm.hpp"
#include "AbstractAlgorithmFactory.hpp"
#include "InteractiveMode.hpp"

#define DESIRED_ADDRESS "127.0.0.1"
#define DESIRED_PORT 3500

/// https://stackoverflow.com/questions/1706551/parse-string-into-argv-argc
int makeargs(char *args, int *argc, char ***aa) {
    char *buf = strdup(args);
    int c = 1;
    char *delim;
    char **argv = (char**) calloc(c, sizeof (char *));

    argv[0] = buf;

    while (delim = strchr(argv[c - 1], ' ')) {
        argv = (char**) realloc(argv, (c + 1) * sizeof (char *));
        argv[c] = delim + 1;
        *delim = 0x00;
        c++;
    }

    *argc = c;
    *aa = argv;

    return c;
}

void runInteractiveMode(AbstractAlgorithmFactory *factory, InteractiveMode *interactiveMode, ParseParams *parse, ParamsCarrier *singleTone)
{
	struct sockaddr_in addr = {0};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(DESIRED_PORT); /*converts short to
										short with network byte order*/
	addr.sin_addr.s_addr = inet_addr(DESIRED_ADDRESS);

	int sock = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (sock == -1) {
		perror("Socket creation error");
		return;
	}
	if (bind(sock, (struct sockaddr*) &addr, sizeof(addr)) == -1) {
		perror("Bind error");
		close(sock);
		return;
	}

	if (listen(sock, 1/*length of connections queue*/) == -1) {
		perror("Listen error");
		close(sock);
		return;
	}

	int client_sock = accept(sock, NULL, NULL); /* 2nd and 3rd argument may be NULL. */
	if (client_sock == -1) {
		perror("Accept error");
		close(sock);
		return;
	}

	kill(getppid(), SIGUSR1);
	char buf[1024];
	do
	{
		bzero(buf, 1024);
		ssize_t readden = recv(sock, buf, 1024, 0);
		if (readden < 0) {
			close(sock);
			return; 
		}
		char **argv;
   	 	int argc;
		if (parse->parseParams(argc, argv) != 1)
		{
			close(sock);
			return;
		}
		singleTone = parse->getParams();
		AbstractAlgorithm *actualAlgorithm;
		actualAlgorithm = factory->getAlgorithm(singleTone->getString("algorithm", "FWMethod"), interactiveMode);
		actualAlgorithm->runAlgorithm(singleTone);
		kill(getppid(), SIGUSR2);
	} while (singleTone->getInt("quit", 1));
}

int main(int argc, char **argv)
{
	InteractiveMode *interactiveMode;
	AbstractAlgorithmFactory *factory = AbstractAlgorithmFactory::CreateFactory(AbstractAlgorithmFactory::TYPE_ALGORITHM::COSMIC);
	ParseParams *parse = new ParseParams();
	ParamsCarrier *singleTone;
	if (parse->parseParams(argc, argv) != 1)
	{
		return -1;
	}
	singleTone = parse->getParams();
	if (singleTone->getInt("interactive", 0))
	{
		interactiveMode = new InteractiveMode();
		runInteractiveMode(factory, interactiveMode, parse, singleTone);
		return 0;
	}
	AbstractAlgorithm *actualAlgorithm;
	actualAlgorithm = factory->getAlgorithm(singleTone->getString("algorithm", "FWMethod"), NULL);
	actualAlgorithm->runAlgorithm(singleTone);
	return 0;
}