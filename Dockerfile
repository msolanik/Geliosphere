FROM nvidia/cuda:10.1-devel

RUN apt update
RUN apt install -y cmake && apt clean

COPY . /cuda_implementation
RUN cd /cuda_implementation && cmake --clean-first -DCMAKE_BUILD_TYPE=Release --install ./ && make

CMD ["bash"]
