"""
    This script is used for batch-run of Geliosphere for input values 
"""
from tomlkit import document, table
import argparse
import csv
import os
import subprocess
import sys

"""
    Generate input settings file based on input r and theta. 
"""
def generate_toml_file(r, theta):
    toml = document()
    defaultValues = table()
    defaultValues.add('K0', 5e22)
    defaultValues.add('V', 400.0)
    defaultValues.add('dt', 50.0)
    defaultValues.add('theta_injection', float(theta))
    defaultValues.add('r_injection', float(r))
    solarPropLikeModelSettings = table()
    solarPropLikeModelSettings.add('SolarProp_ratio', 0.02)
    geliosphereModelSettings = table()
    geliosphereModelSettings.add('Geliosphere_ratio', 0.2)
    geliosphereModelSettings.add('K0_ratio', 5.0)
    geliosphereModelSettings.add('C_delta', 8.7e-5)
    geliosphereModelSettings.add('default_tilt_angle', 0.1)
    advancedSettings = table()
    advancedSettings.add('remove_log_files_after_simulation', True)
    toml.add('default_values', defaultValues)
    toml.add('SolarProp_like_model_settings', solarPropLikeModelSettings)
    toml.add('Geliosphere_model_settings', geliosphereModelSettings)
    toml.add('advanced_settings', advancedSettings)
    with open('./Settings_batch.toml', mode = 'w') as file:
        file.write(toml.as_string())

"""
    Run Geliosphere for every row in input csv file.
"""
def geliophere_batch_run(input_file, geliosphere_executable):
    # Input file must contains items in following order: year,month,day,angle,phi,r,theta
    with open(input_file,mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            generate_toml_file(row['r'], row['theta'])
            targetDirectory = row['year'] + '_' + row['month'] + '_' + row['day']
            pathToSettings = os.path.join(os.curdir, 'Settings_batch.toml') 
            subprocess.run([geliosphere_executable, '-T', '-c', '-s', pathToSettings, '-d', '1000', 
                            '-m', row['month'], '-y', row['year'], '-p', targetDirectory])

"""
    Run simple utility for batch run of Geliosphere.
"""
def main(argv):
    argParser = argparse.ArgumentParser(description='Simple utility for batch run of Geliosphere')
    argParser.add_argument('-g', '--geliosphere-executable', help='Path to Geliosphere executable', default='../build/Geliosphere')
    argParser.add_argument('-i', '--input-csv-file', help='Path to input csv file', default='solarprop_input_paramaters_1AU_theta_90.csv')
    
    args = argParser.parse_args(argv);

    geliophere_batch_run(args.input_csv_file, args.geliosphere_executable)

if __name__ == "__main__":
    main(sys.argv[1:])