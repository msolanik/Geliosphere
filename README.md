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