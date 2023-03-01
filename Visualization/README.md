# Scripts for visualizations
All scripts in this directory were tested with python 3.8.10, recommended version of python is at least >= 3.6. 

Following command will install pipenv, which is necessary for run of scripts: 

```
pip install pipenv
```

After that, following commands will install dependencies and environment for run of scripts:

```
pipenv install
pipenv shell
```

Following files were are based on data from Ulysses: 
```
ulytrj_helio_fmt.txt
ulytrj_helio.asc.txt
KET_P116_P190_1994-1998.csv
```

## create_ulysses_Geliosphere_flux.py
Utility to replicate figure comparing Ulysses trajectory and Geliosphere 2D model results between 1994 and 1998. Script requires to has built-up Geliosphere in ```../build/``` directory.
After success run, flux_KET_P190_vs_SOLARPROPlike_1xdelta0.png will contain mentioned figure.

```
python3 create_ulysses_Geliosphere_flux.py
```

## prepare_input_based_on_ulysses.py
This script is used to prepare input for visualization script from Ulysses trajectory data.  

```
python3 prepare_input_based_on_ulysses.py -i ulytrj_helio.asc.txt -o ulytrj.csv -s ../SolarProp_K0_phi_table.csv -g geliosphere_batch_input_paramaters.csv
```

| Parameter |  Value | Description |
| :--- | :----: | ---: |
| -i | string | Input file with Ulysses trajectories |
| -o | string | Output name for Ulysses trajectory data | 
| -s | string | Path to csv file containing SolarProp input data |
| -g | string | Output file generated for batch run of Geliosphere | 

## batch_run_geliosphere.py
Simple utility for batch run of Geliosphere. Following example demonstrates run of script for solarprop_input_paramaters_1AU_theta_90.csv and Geliosphere build in build directory in root folder of Geliosphere. 

```
python3 batch_run_geliosphere.py -i solarprop_input_paramaters_1AU_theta_90.csv -g ../build/Geliosphere -s 1997 -e 1998
```

| Parameter |  Value | Description |
| :--- | :----: | ---: |
| -g | string | Path to Geliosphere executable |
| -i | string | Path to input csv file | 
| -s | string | Start year (included) |
| -e | string | End year (included) | 

Input csv should contain following values:

```
year,month,day,r,theta
```

## prepare_spectra.py
This script is used for process spectra from Geliosphere for further visualization. Output files contains intensity for each year and bin. 

```
python3 prepare_spectra.py -i ./results -o bins_per_years_output
```

| Parameter |  Value | Description |
| :--- | :----: | ---: |
| -i | string | Path to Geliosphere results directory |
| -o | string | Path to output directory | 

## create_plot.py
This script is responsible for visualizing Ulysses and Geliosphere energetic spectra.

```
python3 create_plot.py -u ulytrj.csv -f KET_P116_P190_1994-1998.csv -i bins_per_years_output/0.2500.csv -t P190 -o flux_KET_P190_vs_SOLARPROPlike_1xdelta0
```

| Parameter |  Value | Description |
| :--- | :----: | ---: |
| -u | string | Path to file with Ulysses trajectories |
| -f | string | Path to Ulysses flux file | 
| -i | string | Path to input csv file from Geliosphere containing data for single bin | 
| -t | string | Type of the flux from ulysses | 
| -o | string | Output image name | 
