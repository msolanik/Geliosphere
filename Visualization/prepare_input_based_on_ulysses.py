"""
    This script is used to prepare input for visualization script from Ulysses trajectory data.  
"""
import argparse
import csv
import datetime
import re
import sys

"""
    Run utility for preparation of data for visualization script from Ulysses trajectory data.
"""
def main(argv):
    argParser = argparse.ArgumentParser(description='Utility preparing data for visualization script from Ulysses trajectory data')
    argParser.add_argument('-i', '--input-file-name', help='Input file with Ulysses trajectories', default='ulytrj_helio.asc.txt')
    argParser.add_argument('-o', '--output-file-name', help='Output name for Ulysses trajectory data', default='ulytrj.csv')
    argParser.add_argument('-s', '--solarprop-input-parameters-file', help='Path to csv file containing SolarProp input data', default='../Geliosphere_K0_phi_table.csv')
    argParser.add_argument('-g', '--geliosphere-batch-file', help='Output file generated for batch run of Geliosphere', default='geliosphere_batch_input_paramaters.csv')
    args = argParser.parse_args(argv);
    
    ulyssesData = []
    with open(args.input_file_name,mode = 'r') as file:
        lines = file.readlines()
        for line in lines:
            columns = re.split(r'\s+', line)
            # column[0] -> IYEAR
            # column[1] -> IDAY
            # column[8] -> HRANGE
            # column[11] -> HECLAT
            # column[12] -> SOLONG
            ulyssesData.append([columns[0], columns[1], columns[8], columns[11], columns[12]])
    header = ['year','day_of_year','r_range','trajectory_latitude','trajectory_longitude']
    with open(args.output_file_name, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(ulyssesData)

    availableDates = dict()
    with open(args.solarprop_input_parameters_file, 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            dayInYear = datetime.datetime(int(row['year']), int(row['month']), int(row['day'])).timetuple().tm_yday
            availableDates[str(dayInYear) + '_' + row['year']] = { "day": row['day'], "month": row['month'], "year": row['year'] } 
        
    header = ['year','month','day','r','theta']
    with open(args.geliosphere_batch_file, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for trajectoryData in ulyssesData:
            dayOfYearAndYear = trajectoryData[1] + '_' + trajectoryData[0]
            if dayOfYearAndYear in availableDates:
                writer.writerow([availableDates[dayOfYearAndYear]['year'], availableDates[dayOfYearAndYear]['month'], availableDates[dayOfYearAndYear]['day'], trajectoryData[2], 90.0 - float(trajectoryData[3])])

if __name__ == "__main__":
    main(sys.argv[1:])