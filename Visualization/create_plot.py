"""
    This script is responsible for visualizing Ulysses and Geliosphere energetic spectra 
"""
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

"""
    latitude mark - to show when Ulysses was on low latitudes close to ecliptic.
    Input csv file must contain following columns: year,day_of_year,r_range,trajectory_latitude,trajectory_longitude 
"""
def visualize_ulysses_low_latitudes(path_file_ulysses_trajectories, plt):
    yy = []
    yy.append(0)
    yy.append(0.0003)
    with open(path_file_ulysses_trajectories,mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            year = int(row['year'])
            if year >= 1994 and year < 1999:
                trajectory_latitude = float(row['trajectory_latitude'])
                if trajectory_latitude<10. and trajectory_latitude>-10.:
                    plt.plot([year + (float(row['day_of_year']) / 365.25)] * 2,yy,color="grey",alpha=0.01)
                if (trajectory_latitude>10. and trajectory_latitude<20.) or (trajectory_latitude<-10. and trajectory_latitude>-20.):
                    plt.plot([year + (float(row['day_of_year']) / 365.25)] * 2,yy,color="yellow",alpha=0.02)

"""
    Visualize Ulysses flux.
    Input csv file must contain following columns: year,day_of_year,P116_flux,P190_flux 
"""
def visualize_ulysses_flux(path_file_ulysses_fluxes, flux_type, plt):
    ulysses_time = []
    flux = []
    with open(path_file_ulysses_fluxes,mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            ulysses_time.append(int(row['year']) + (int(row['day_of_year']) / 365.0))
            flux.append(float(row['P190_flux' if flux_type == r'P190' else "P116_flux"]))
    plt.scatter(ulysses_time,flux,color="green", label="Ulysses KET " + flux_type +" flux")

"""
    Visualize Geliosphere flux.
    Input csv file must contain following columns: doy,SpeAvg 
"""
def visualize_geliosphere_flux(path_to_geliosphere_file, plt):
    time = []
    flux = []
    with open(path_to_geliosphere_file,mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            time.append(float(row['doy']))
            flux.append(float(row['SpeAvg']) / (10000000.0))
    plt.scatter(time,flux,color="red", label="Geliosphere 2D model P190 simulation")

"""
    Run simple utility to visualize Ulysses and Geliosphere energetic spectra.
"""
def main(argv):
    argParser = argparse.ArgumentParser(description='Simple utility to visualize Ulysses and Geliosphere energetic spectra')
    argParser.add_argument('-u', '--ulysses-trajectory-file', help='Path to file with Ulysses trajectories', default='ulytrj.csv')
    argParser.add_argument('-f', '--ulysses-flux-file', help='Path to Ulysses flux file', default='KET_P116_P190_1994-1998.csv')
    argParser.add_argument('-i', '--input-csv-file', help='Path to input csv file from Geliosphere containing data for single bin', default='0.2500.csv')
    argParser.add_argument('-t', '--flux-type', help='Type of the flux from ulysses', choices = ['P190', 'P116'], default='P190')
    argParser.add_argument('-o', '--output-image-name', help='Output image name', default='flux_KET_P190_vs_SOLARPROPlike_1xdelta0')
    
    args = argParser.parse_args(argv);

    mpl.use("pgf")
    # Adding siunitx package based on:
    # https://tex.stackexchange.com/questions/391074/how-to-use-the-siunitx-package-within-python-matplotlib
    pgf_with_latex = {                      
        "pgf.texsystem": "pdflatex",       
        "text.usetex": True,               
        "font.family": "serif",
        "font.serif": [],                   
        "font.sans-serif": [],             
        "font.monospace": [],
        "axes.labelsize": 14,              
        "font.size": 14,
        "legend.fontsize": 14,               
        "xtick.labelsize": 14,              
        "ytick.labelsize": 14,
        "figure.figsize": [10,6],     
        "pgf.preamble": "\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,locale=UK]{siunitx}",
            r"""\ifdefined\qty\else
                \ifdefined\NewCommandCopy
                    \NewCommandCopy\qty\SI
                \else
                    \NewDocumentCommand\qty{O{}mm}{\SI[#1]{#2}{#3}}
                \fi
                \fi
                \ifdefined\unit\else
                \ifdefined\NewCommandCopy
                    \NewCommandCopy\unit\si
                \else
                    \NewDocumentCommand\unit{O{}m}{\si[#1]{#2}}
                \fi
                \fi"""
            ])
    }
    plt.rcParams.update(pgf_with_latex)

    fig = plt.figure(figsize=(10, 6), dpi=300)
    
    visualize_ulysses_low_latitudes(args.ulysses_trajectory_file, plt)
    visualize_ulysses_flux(args.ulysses_flux_file, args.flux_type, plt)
    visualize_geliosphere_flux(args.input_csv_file, plt)
    
    plt.xlabel("time [year]")
    # ^2 did showed up properly
    plt.ylabel(r'flux [\unit{1/(s . cm\textsuperscript{2} . sr . MeV / n)}]')
    title = 'Comparison of Ulysses KET P190 flux and Geliosphere 2D model P190 simulation'
    plt.title(title)
    plt.legend(loc='upper right')
    fig.savefig(args.output_image_name+ ".png", dpi = 300)

if __name__ == "__main__":
    main(sys.argv[1:])