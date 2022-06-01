FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

RUN apt-get update -y 
RUN apt-get -y install git
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y cmake && apt-get clean

COPY . /cuda_implementation
RUN cd /cuda_implementation && cmake --clean-first -DCMAKE_BUILD_TYPE=Release --install ./ && make

CMD ["bash"]
