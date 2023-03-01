"""
    Following script is demonstrating visualization of comparison between Geliosphere and Ulysses fluxes between 1994-1999.
"""

import os

"""
    Prepare trajectories from Ulysses and input file for batch run for further processing.
"""
os.system(r"python3 prepare_input_based_on_ulysses.py -i ulytrj_helio.asc.txt -o ulytrj.csv -s ../SolarProp_K0_phi_table.csv -g geliosphere_batch_input_paramaters.csv")

"""
    Geliosphere batch run for dates between 1994 and 1998. 
"""
os.system(r"python3 batch_run_geliosphere.py -g ../build/Geliosphere -i geliosphere_batch_input_paramaters.csv -s 1994 -e 1998")

"""
    Prepare spectra for visualization.
"""
os.system(r"python3 prepare_spectra.py -i ./results -o bins_per_years_output")

"""
    Visualize energetic spectra from Ulysses and Geliosphere.
"""
os.system(r"python3 create_plot.py -u ulytrj.csv -f KET_P116_P190_1994-1998.csv -i bins_per_years_output/0.2500.csv -t P190 -o flux_KET_P190_vs_SOLARPROPlike_1xdelta0")