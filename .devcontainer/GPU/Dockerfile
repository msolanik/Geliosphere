FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu

RUN apt-get update -y 
RUN apt-get -y install gdb
RUN apt-get -y install git
RUN apt-get -y install g++
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y cmake && apt-get clean