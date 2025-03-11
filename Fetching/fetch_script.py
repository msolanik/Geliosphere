# @file fetch_script.py  
# @author: Marek Hamracek
#
#
#   SETTING UP CRON
#
#       crontab -e
#       0 10 1 * * /path/to/python3 /path/to/fetch_script.py    #First day of any month at 10 AM
#                                                               #Replace /path/to/
#
#
#

import requests
import pandas as pd
import urllib3
import datetime
import argparse

# Disable the InsecureRequestWarning - http://wso.stanford.edu/Tilts.html site does not have a certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define the URL
url_flow_speed = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
url_phi = "https://cosmicrays.oulu.fi/phi/Phi_Table_2017.txt"
url_tilt_angle = "http://wso.stanford.edu/Tilts.html"

# Define the payload for OMNIWEB form request
payload = {
    "activity": "retrieve",
    "res": "daily",          
    "spacecraft": "omni2_daily",  
    "start_date": "20250101",  
    "end_date": "20250110",    
    "vars": "24", 
}

def connect_to_url(url, method='get', data=None, verify_ssl=True):
    """Connect to a URL and handle exceptions."""
    try:
        if method == 'get':
            response = requests.get(url, verify=verify_ssl)
        elif method == 'post':
            response = requests.post(url, data=data)
        response.raise_for_status()
        print(f"{url} connected successfully.")
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to {url}: {e}")
        return None

def parse_to_csv(html_content: str, table_start: str, table_end: str, header_start: str, output_file_path: str):
    """Parse HTML content and save to CSV."""

    try:
        # Clean text_data
        text_data = html_content.replace("\x00", "")
        start_index, end_index = None, None
        header = None
        lines = text_data.splitlines()

        # Locate to-be-read table on html content
        for i, line in enumerate(lines):
            if header_start in line:
                header = line.split()

            if table_end in line and start_index is not None:
                end_index = i
                break
            elif table_start in line:
                start_index = i

        if start_index is None or end_index is None:
            raise ValueError("Table boundaries not found.\n\n")            

        if header is None:
            print("Warning: No header found, generating default header.")

        # Convert to desired format
        table_string = "\n".join(lines[start_index + 1:end_index])
        rows = table_string.split("\n")
        data = [row.split() for row in rows]

        # Write to .csv file
        df = pd.DataFrame(data, columns=header)  
        df.to_csv(output_file_path, index=False)
        print("Table data were saved to:", output_file_path)

        with open(args.log_file, "a") as log_file:
            log_file.write("\nThe most up-to-date entry from " + output_file_path + ":\n" + str(df.tail(1)))

        return df.tail(1)

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unspecified error occurred: {e}")

def compute_new_entry(last_entry_flow_speed, last_entry_phi, last_entry_tilt_angle):
    """Compute and append a new entry to an existing CSV."""

    filepath = args.csv_file

    try:
        df = pd.read_csv(filepath)

        # Date format conversion
        date = pd.to_datetime(last_entry_tilt_angle['Start'], format='%Y:%m:%d')

        carrington_rotation = last_entry_tilt_angle.iloc[0]['Rot']
        year_week = round(date.dt.year.iloc[0] + date.dt.day_of_year.iloc[0] / 365, 7)
        modulation_potentional = last_entry_phi.iloc[0].dropna().iloc[-1] # Last non-NaN value #Geliosphere has some calcs that are yet to be considered
        k0_au2_s = (float(last_entry_flow_speed.iloc[0]['1']) * 6.68459e-9) * (99 / (3 * int(modulation_potentional))) #1D Models
        tilt_angle = last_entry_tilt_angle.iloc[0]['R_av']
        year = date.dt.year.iloc[0]
        month = date.dt.month.iloc[0]
        day = date.dt.day.iloc[0]
        #polarity = 1 if (month in [12, 1, 2] and ((month == 12 and day >= 21) or month == 1 or month == 2)) or month in [3, 4, 5, 10, 11] else -1
        reference_year = 2012
        years_since_ref = year - reference_year

        # Determine polarity by checking the 11-year cycle
        if (years_since_ref // 11) % 2 == 0:
            polarity = 1
        else:
            polarity = -1

        # 2D Solarprop-like Models
        #if polarity == 1:
        #    k0_au2_s = (137/int(modulation_potentional)) - 0.061
        #elif polarity == -1:
        #    k0_au2_s = 0.07 * (137/int(modulation_potentional)) - 0.061
        

        # Create a new row as a DataFrame
        new_row = pd.DataFrame({
            'carrington_rotation': [carrington_rotation],
            'year_week': [year_week],
            'modulation_potentional': [modulation_potentional],
            'k0_au2/s': [k0_au2_s],
            'tilt_angle': [tilt_angle],
            'year': [year],
            'month': [month],
            'day': [day],
            'polarity': [polarity]
        })

        df = pd.concat([df, new_row], ignore_index=True)
       
        print("\n Last entries: file: ",filepath," \n---------------------------------------------------")
        print(df.tail(5),"\n")
        df.to_csv(filepath, index=False)

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An error occurred when computing new entry: {e}")

def log_script_start():
    """Log the start of the script."""

    with open(args.log_file, "w") as log_file:
        log_file.write(f"\nScript started successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    """Main function to orchestrate the script."""

    print("\n")
    print(f"Script started successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")

    log_script_start()

    response_flow_speed = connect_to_url(url_flow_speed, method='post', data=payload)
    response_phi = connect_to_url(url_phi, verify_ssl=False)
    response_tilt_angle = connect_to_url(url_tilt_angle)

    if response_flow_speed and response_phi and response_tilt_angle:
        last_entry_flow_speed = parse_to_csv(response_flow_speed.text, "YEAR DOY", "</pre>", "YEAR DOY", "table_data_flow_speed.csv")
        last_entry_phi = parse_to_csv(response_phi.text, "+++", "+++", "Year  Jan", "table_data_phi.csv")
        last_entry_tilt_angle = parse_to_csv(response_tilt_angle.text, "Carr Rot", "</pre>", "Carr Rot", "table_data_tilt_angle.csv")
        compute_new_entry(last_entry_flow_speed, last_entry_phi, last_entry_tilt_angle)

        print(args.csv_file, "has been updated successfully")

    print(args.log_file ," has been updated successfully\n")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch and process data.")
    parser.add_argument('--csv_file', type=str, default='Geliosphere_K0_phi_table.csv', help="Path to the output CSV file")
    parser.add_argument('--log_file', type=str, default='fetch_script_log.txt', help="Path to the output log file")

    args = parser.parse_args()

    main()