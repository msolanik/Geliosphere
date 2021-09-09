# Geliosphere

Geliosphere is application that utilize GPU as a computing unit for cosmic ray modulation in Heliosphere. Contains 1D models of cosmic ray modulation in heliosphere based on [this paper.](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2015JA022237)
  

| Parameter | Value | Description |
| :--- | :----: | ---: |
| -F | - | Run forward-in-time simulation |
| -B | - | Run backward-in-time simulation |
| -c | - | Set .csv output |
| -d | float | Set time step (default value 5.0s) |
| -K | float | Set K0 (default value 5∗10^22cm2/s) |
| -V | float | Set solar wind speed (default value 400 km/s)|
| -p | string | Set custom path for output in output directory |
| -N | int | Set amount of simulations in millions |

Only GPUs from Nvidia Pascal and Turing architectures are supported. List of supported GPUs:

> Nvidia T4
> Nvidia P100
> Tesla P100
> Tesla P40
> Tesla P4
> Geforce GTX 1650 Ti
> Nvidia Titan RTX
> Geforce RTX 2080Ti
> Geforce RTX 2080
> Geforce RTX 2070
> Geforce RTX 2060
> Nvidia Titan Xp
> Nvidia Titan X
> Geforce GTX 1080 Ti
> Geforce GTX 1080
> Geforce GTX 1070 Ti
> Geforce GTX 1070
> Geforce GTX 1060
> Geforce GTX 1050

Additional information about used models can be found in following articles:

[Accuracy and comparasion results from GPU implementation to Crank-Nicolson model](https://pos.sissa.it/395/1320/pdf)