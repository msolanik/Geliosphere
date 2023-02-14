# Geliosphere

Geliosphere is application that utilize GPU as a computing unit for cosmic ray modulation in Heliosphere. Contains 1D models of cosmic ray modulation in heliosphere based on [this paper.](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237)

Class documentation can be found [here](https://msolanik.github.io/Geliosphere/annotated.html).
  

| Parameter | Value | Description |
| :--- | :----: | ---: |
| -F | - | Run forward-in-time simulation |
| -B | - | Run backward-in-time simulation |
| -E | - | Run SolarpropLike 2D backward-in-time simulation |
| -T | - | Run Geliosphere 2D backward-in-time simulation |
| -c | - | Set .csv output |
| -h | - | Print help for Geliosphere |
| -d | float | Set time step (default value 5.0s) |
| -K | float | Set K0 (default value 5∗10^22cm2/s) |
| -V | float | Set solar wind speed (default value 400 km/s)|
| -p | string | Set custom path for output in output directory |
| -N | int | Set amount of simulations in millions |
| -m | int | Set month for using meassured values |
| -y | int | Set year for using meassured values |
| -s | string | Set path to settings toml file (default Settings.tml in root folder) |

All GPUs from Nvidia Pascal, Amphere, Volta and Turing architectures are supported.

Additional information about used models can be found in following articles:

[Accuracy and comparasion results from GPU implementation to Crank-Nicolson model](https://pos.sissa.it/395/1320/pdf)

## Installation
<details>

### Geliosphere with GPU support
Standard installation of the GPU version of Geliosphere requires installation of the Nvidia toolkit, g++ and cmake 3.14+. These packages can be installed via any packaging tool. The following example is provided for the apt-get packaging tool:
  ```
  sudo apt-get install cuda g++ cmake
  ```

Different Linux distributions may have different approach for CUDA installation.

After installation is complete, an optimized version of the tool can be built via the following command:
  ```
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  ```

After build is complete successfully, executable is placed in build directory with Geliosphere. For further instruction regarding the program usage, following command will display help for the user:
  ```
  ./build/Geliosphere --help
  ```

### Geliosphere with CPU-only support
The packages are similar, with the exception that the CPU version naturally does not require installation of the Nvidia toolkit. CPU-only version of Geliosphere can be built via the following command:
  ```
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCPU_VERSION_ONLY=1
  cmake --build build
  ```

### Dockerized versions
We also included runner scripts(<em>runner.sh</em> and <em>runner_cpu_only.sh</em>), that can build and run Geliosphere in Docker. They automatically build Docker image, however it can re-built via:

  ```
  ./runner.sh -f
  ./runner_cpu_only.sh -f
  ```

Help for Geliosphere can be displayed via following command:
  ```
  ./runner.sh --help
  ./runner_cpu_only.sh --help
  ```
</details>

## Module structure
<details>

Following image describes relations between modules in Geliosphere:

![module_diagram drawio (1)](https://user-images.githubusercontent.com/22960818/200040994-dde685a6-9990-4a78-83e6-0975eaf230f3.png)

Modules are used to organize the logic needed for simulations in the heliosphere and to support logic for them. These modules are described as follows: 
- **Geliosphere** - contains the main function and links basic logic for selecting the model, parsing input data and running the selected model,
- **Algorithm** - contains logic used for selecting implementation of model for selected computing unit, and logic for analyzing output spectra, 
- **Factory** - contains classes based on factory and abstract factory patterns used for creating objects,  
- **Constants** - contains classes used for parsing input data,
- **CPU Implementations** - contains classes used for running parallel CPU implementations of models of cosmic rays modulation in the heliosphere,
- **CUDA Kernel** - contains classes used for running parallel GPU implementations of models of cosmic rays modulation in the heliosphere,
- **Utils** - contains classes holding various functions used in Geliosphere.
</details>

## Source file description
<details>

### Geliosphere module
<details>

```
Geliosphere
|
│    Dockerfile
|    Dockerfile.CPU  
|    main.cpp
└───Algorithm
└───Constants
└───CpuImplementations
└───CUDAKernel
└───Factory
└───Utils
```

<strong>Geliosphere</strong> module contains following source files:

- <strong>Dockerfile</strong> - file containing defitinion for building GPU Docker image with GPU support.
- <strong>Dockerfile.CPU</strong> - file containing defitinion for building GPU Docker image with CPU-only support.
- <strong>main.cpp</strong> - file containing main functions with needed iteractions between modules. 

</details>

### Algorithm module

<details>

```
Algorithm
│    
└───include
|   |   AbstractAlgorithm.hpp
|   |   OneDimensionBpAlgorithm.hpp
|   |   OneDimensionBpResults.hpp
|   |   OneDimensionFpAlgorithm.hpp
|   |   OneDimensionFpResults.hpp
|   |   ResultConstants.hpp
|   |   ThreeDimensionBpAlgorithm.hpp
|   |   TwoDimensionBpAlgorithm.hpp
|   |   TwoDimensionBpResults.hpp
└───src
    |   AbstractAlgorithm.cpp
    |   OneDimensionBpAlgorithm.cpp
    |   OneDimensionBpResults.cpp
    |   OneDimensionFpAlgorithm.cpp
    |   OneDimensionFpResults.cpp
    |   ThreeDimensionBpAlgorithm.cpp
    |   TwoDimensionBpAlgorithm.cpp
    |   TwoDimensionBpResults.cpp
```


<strong>Algorithm</strong> module contains following source files:

- <strong>AbstractAlgorithm.hpp</strong> - Header file of abstract definition for algorithm.
- <strong>OneDimensionBpAlgorithm.hpp</strong> - Header file of implementation of 1D B-p method
- <strong>OneDimensionBpResults.hpp</strong> - Header file of implementation of 1D B-p method analyzer for output data.
- <strong>OneDimensionFpAlgorithm.hpp</strong> - Header file of implementation of 1D F-p method
- <strong>OneDimensionFpResults.hpp</strong> - Header file of implementation of 1D F-p method analyzer for output data.
- <strong>ResultConstants.hpp</strong> - Header file containing constants needed for analysis of log files for all models.
- <strong>ThreeDimensionBpAlgorithm.hpp</strong> - Header file of implementation of Geliosphere 2D B-p method.
- <strong>TwoDimensionBpAlgorithm.hpp</strong> - Header file of implementation of SolarProp-like 2D B-p method.
- <strong>TwoDimensionBpResults.hpp</strong> - Header file of implementation of 2D B-p method analyzer for output data.

- <strong>AbstractAlgorithm.cpp</strong> - Source file of abstract definition for algorithm.
- <strong>OneDimensionBpAlgorithm.cpp</strong> - Source file of implementation of 1D B-p method
- <strong>OneDimensionBpResults.cpp</strong> - Source file of implementation of 1D B-p method analyzer for output data.
- <strong>OneDimensionFpAlgorithm.cpp</strong> - Source file of implementation of 1D F-p method
- <strong>OneDimensionFpResults.cpp</strong> - Source file of implementation of 1D F-p method analyzer for output data.
- <strong>ThreeDimensionBpAlgorithm.cpp</strong> - Source file of implementation of Geliosphere 2D B-p method.
- <strong>TwoDimensionBpAlgorithm.cpp</strong> - Source file of implementation of SolarProp-like 2D B-p method.
- <strong>TwoDimensionBpResults.cpp</strong> - Source file of implementation of 2D B-p method analyzer for output data.

</details>

### Factory module

<details>

```
Factory
│    
└───include
|   |   AbstractAlgorithmFactory.hpp
|   |   CosmicFactory.hpp
└───src
    |   AbstractAlgorithmFactory.cpp
    |   CosmicFactory.cpp
```

<strong>Factory</strong> module contains following source files:

- <strong>AbstractAlgorithmFactory.hpp</strong> - Interface of Abstract Factory Pattern.
- <strong>CosmicFactory.hpp</strong> - Class represents implementation of Factory Pattern for cosmic algorithms.
- <strong>AbstractAlgorithmFactory.cpp</strong> - Source file for interface of Abstract Factory Pattern.
- <strong>CosmicFactory.cpp</strong> - Source file of class represents implementation of Factory Pattern for cosmic algorithms.

</details>

### Constants module

<details>

```
Constants
│    
└───include
|   |   MeasureValuesTransformation.hpp
|   |   ParamsCarrier.hpp
|   |   ParseParams.hpp
|   |   TomlSettings.hpp
└───src
    |   MeasureValuesTransformation.cpp
    |   ParamsCarrier.cpp
    |   ParseParams.cpp
    |   TomlSettings.cpp
```

<strong>Constants</strong> module contains following source files:

- <strong>MeasureValuesTransformation.hpp</strong> - Header file for class representing extraction of measured parameters for simulation from table.
- <strong>ParamsCarrier.hpp</strong> - Header file for universal map-like structure.
- <strong>ParseParams.hpp</strong> - Header file of parser of arguments from CLI
- <strong>TomlSettings.hpp</strong> - Header file for class representing parser of values from settings.
- <strong>MeasureValuesTransformation.cpp</strong> - Source file for class representing extraction of measured parameters for simulation from table.
- <strong>ParamsCarrier.cpp</strong> - Source file for universal map-like structure.
- <strong>ParseParams.cpp</strong> - Source file of parser of arguments from CLI
- <strong>TomlSettings.cpp</strong> - Source file for class representing parser of values from settings.

</details>

### CPU Implementations module

<details>

```
CpuImplementations
│    
└───include
|   |   AbstractCpuSimulation.hpp
|   |   Constants.hpp
|   |   OneDimensionBpCpuSimulation.hpp
|   |   OneDimensionFpCpuSimulation.hpp
|   |   ThreeDimensionBpCpuSimulation.hpp
|   |   TwoDimensionBpCpuSimulation.hpp
└───src
    |   OneDimensionBpCpuSimulation.cpp
    |   OneDimensionFpCpuSimulation.cpp
    |   ThreeDimensionBpCpuSimulation.cpp
    |   TwoDimensionBpCpuSimulation.cpp
```

<strong>CPU Implementations</strong> module contains following source files:

- <strong>AbstractCpuSimulation.hpp</strong> - Abstract definition for implementation of model on CPU.
- <strong>Constants.hpp</strong> - Header file for constants for CPU implementations.
- <strong>OneDimensionBpCpuSimulation.hpp</strong> - Header file for CPU implementation for 1D B-p model.
- <strong>OneDimensionFpCpuSimulation.hpp</strong> - Header file for CPU implementation for 1D F-p model.
- <strong>ThreeDimensionBpCpuSimulation.hpp</strong> - Header file for CPU implementation for Geliosphere 2D B-p model.
- <strong>TwoDimensionBpCpuSimulation.hpp</strong> - Header file for CPU implementation for SolarProp-like 2D B-p model.

- <strong>OneDimensionBpCpuSimulation.cpp</strong> - Source file for CPU implementation for 1D B-p model.
- <strong>OneDimensionFpCpuSimulation.cpp</strong> - Source file for CPU implementation for 1D F-p model.
- <strong>ThreeDimensionBpCpuSimulation.cpp</strong> - Source file for CPU implementation for Geliosphere 2D B-p model.
- <strong>TwoDimensionBpCpuSimulation.cpp</strong> - Source file for CPU implementation for SolarProp-like 2D B-p model.
  
</details>

### CUDA Kernel module

<details>

```
CUDAKernel
│    
└───include
|   |   AbstractGpuSimulation.hpp
|   |   CosmicConstants.cuh
|   |   CosmicUtils.cuh
|   |   CudaErrorCheck.cuh
|   |   OneDimensionBpGpuSimulation.hpp
|   |   OneDimensionBpSimulation.cuh
|   |   OneDimensionFpGpuSimulation.hpp
|   |   OneDimensionFpSimulation.cuh
|   |   ThreeDimensionBpGpuSimulation.hpp
|   |   ThreeDimensionBpSimulation.cuh
|   |   TwoDimensionBpGpuSimulation.hpp
|   |   TwoDimensionBpSimulation.cuh
└───src
    |   CosmicConstants.cu
    |   CosmicUtils.cu
    |   OneDimensionBpGpuSimulation.cpp
    |   OneDimensionBpSimulation.cu
    |   OneDimensionFpGpuSimulation.cpp
    |   OneDimensionFpSimulation.cu
    |   ThreeDimensionBpGpuSimulation.cpp
    |   ThreeDimensionBpSimulation.cu
    |   TwoDimensionBpGpuSimulation.cpp
    |   TwoDimensionBpSimulation.cu
```

<strong>CUDA Kernel</strong> module contains following source files:

- <strong>AbstractGpuSimulation.hpp</strong> - Abstract definition for implementation of model on GPU.
- <strong>CosmicConstants.cuh</strong> - Header file for constants needed for simulations.
- <strong>CosmicUtils.cuh</strong> - Header file for common functions for simulations.
- <strong>CudaErrorCheck.cuh</strong> - Header file for utilities for checking errors.
- <strong>OneDimensionBpGpuSimulation.hpp</strong> - Header file for class utilizing GPU implementation of 1D B-p model.
- <strong>OneDimensionBpSimulation.cuh</strong> - Header file for GPU implementation of 1D B-p model.
- <strong>OneDimensionFpGpuSimulation.hpp</strong> - Header file for class utilizing GPU implementation of 1D F-p model.
- <strong>OneDimensionFpSimulation.cuh</strong> - Header file for GPU implementation of 1D F-p model.
- <strong>ThreeDimensionBpGpuSimulation.hpp</strong> - Header file for class utilizing GPU implementation of Geliosphere 2D B-p model.
- <strong>ThreeDimensionBpGpuSimulation.cuh</strong> - Header file for GPU implementation of Geliosphere 2D B-p model.
- <strong>TwoDimensionBpGpuSimulation.hpp</strong> - Header file for class utilizing GPU implementation of SolarProp-like 2D B-p model.
- <strong>TwoDimensionBpSimulation.cuh</strong> - Header file for GPU implementation of SolarProp-like 2D B-p model.

- <strong>CosmicConstants.cu</strong> - Source file for constants needed for simulations.
- <strong>CosmicUtils.cu</strong> - Source file for common functions for simulations.
- <strong>OneDimensionBpGpuSimulation.cpp</strong> - Source file for class utilizing GPU implementation of 1D B-p model.
- <strong>OneDimensionBpSimulation.cu</strong> - Source file for GPU implementation of 1D B-p model.
- <strong>OneDimensionFpGpuSimulation.cpp</strong> - Source file for class utilizing GPU implementation of 1D F-p model.
- <strong>OneDimensionFpSimulation.cu</strong> - Source file for GPU implementation of 1D F-p model.
- <strong>ThreeDimensionBpGpuSimulation.cpp</strong> - Source file for class utilizing GPU implementation of Geliosphere 2D B-p model.
- <strong>ThreeDimensionBpGpuSimulation.cu</strong> - Source file for GPU implementation of Geliosphere 2D B-p model.
- <strong>TwoDimensionBpGpuSimulation.cpp</strong> - Source file for class utilizing GPU implementation of SolarProp-like 2D B-p model.
- <strong>TwoDimensionBpSimulation.cu</strong> - Source file for GPU implementation of SolarProp-like 2D B-p model.

</details>

### Utils module

<details>

```
Utils
│    
└───include
|   |   FileUtils.hpp
|   |   ResultsUtils.hpp
└───src
    |   FileUtils.cpp
    |   ResultsUtils.cpp
```

<strong>Utils</strong> module contains following source files:

- <strong>FileUtils.hpp</strong> - Header file for utilities for manipulating with directories. 
- <strong>ResultsUtils.hpp</strong> - Header file for utilities for analyting log files.
- <strong>FileUtils.cpp</strong> - Source file for utilities for manipulating with directories. 
- <strong>ResultsUtils.cpp</strong> - Source file for utilities for analyting log files.

</details>

</details>