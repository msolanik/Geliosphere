"""
    This script is used for process spectra from Geliosphere for further visualization.  
"""
import argparse
import csv
import glob
import os
import re
import sys

""" 
    If year is a leap year return True
    else return False 
"""
def is_leap_year(year):
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0

""" 
    Given year, month, day return day of year
    Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 
"""
def doy(Y,M,D):
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N

""" 
    Format year to output format. 
"""
def doy_year(year, doy):
    if is_leap_year(year):
        return year + (doy / 366)
    else:
        return year + (doy / 365)

"""
    Create output files for visualizations.
"""
def create_output_files_for_visualization(input_folder, output_folder):
    dictionary = dict()
    for pathToFile in glob.iglob( input_folder + '/**/Ulysses.csv', recursive=True):
        m = re.search(r'.*\/(\d+)_(\d+)_(\d+)\/Ulysses\.csv$', pathToFile)
        doy_current_spectra = doy(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        with open(pathToFile, 'r') as file:
            csvFile = csv.DictReader(file)
            for row in csvFile:
                if row['Tkin'] in dictionary:
                    dictionary[row['Tkin']].append({'doy': doy_year(int(m.group(1)), doy_current_spectra), 'speAvg': row['speAvg']})
                else:
                    dictionary[row['Tkin']] = [{'doy': doy_year(int(m.group(1)), doy_current_spectra), 'speAvg': row['speAvg']}]
    for Tkin, values in dictionary.items():
        values.sort(key = lambda x: x['doy'])
        outputFile = os.path.join(output_folder, str(Tkin) + '.csv') 
        os.makedirs(os.path.dirname(outputFile), exist_ok=True)
        with open(outputFile, 'w') as file:
            file.write('doy,SpeAvg\n')
            for value in values:
                file.write(str(value['doy']) + ',' + value['speAvg'] + '\n')

"""
    Run utility for preparation of spectra from Geliosphere for further visualization.
"""
def main(argv):
    argParser = argparse.ArgumentParser(description='Simple utility for batch run of Geliosphere')
    argParser.add_argument('-i', '--input-directory', help='Path to Geliosphere results directory', default='./results')
    argParser.add_argument('-o', '--output-directory', help='Path to output directory', default='./output')
    args = argParser.parse_args(argv);
    create_output_files_for_visualization(args.input_directory, args.output_directory)

if __name__ == "__main__":
    main(sys.argv[1:])