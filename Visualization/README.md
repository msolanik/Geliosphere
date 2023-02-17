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