# %%
from datetime import datetime
import io

import pandas as pd
import requests


def get_ghcnd_data(station_id, date_str):
    """
    Retrieve GHCN-Daily weather data for a specific station and date
    
    Parameters:
    -----------
    station_id : str
        The GHCN station identifier (e.g., 'USW00094846' for Chicago O'Hare)
    date_str : str
        Date in YYYY-MM-DD format
    
    Returns:
    --------
    pandas.DataFrame containing the weather data
    """
    # Format the date for the NOAA API
    date_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    year = date_obj.strftime("%Y")
    
    # Base URL for the NOAA GHCN-Daily API
    base_url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
    
    # Construct the URL for the specific station and year
    url = f"{base_url}/{year}/{station_id}.csv"
    
    # Make the HTTP request
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(io.StringIO(response.text), 
                        parse_dates=['DATE'])
    
    # Filter the data for the specific date
    result_df = df[df['DATE'] == formatted_date]
    
    if result_df.empty:
        print(f"No data found for station {station_id} on {formatted_date}")
        return None
    
    return result_df

def parse_ghcnd_data(data_df):
    """
    Parse and interpret the GHCN-Daily data
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing GHCN-Daily data
    
    Returns:
    --------
    pandas.DataFrame with interpreted weather data
    """
    if data_df is None or data_df.empty:
        return None
    
    # Create a new DataFrame to store interpreted data
    result = pd.DataFrame()
    
    # Add station ID and date
    result['STATION'] = data_df['STATION']
    result['DATE'] = data_df['DATE']
    
    # Interpret the weather elements
    for index, row in data_df.iterrows():
        element = row['ELEMENT']
        value = row['VALUE']
        
        # Temperature values are stored in tenths of degrees C
        if element == 'TMAX':
            result.loc[index, 'MAX_TEMP_C'] = value / 10.0
            result.loc[index, 'MAX_TEMP_F'] = (value / 10.0) * 9/5 + 32
        elif element == 'TMIN':
            result.loc[index, 'MIN_TEMP_C'] = value / 10.0
            result.loc[index, 'MIN_TEMP_F'] = (value / 10.0) * 9/5 + 32
        elif element == 'TAVG':
            result.loc[index, 'AVG_TEMP_C'] = value / 10.0
            result.loc[index, 'AVG_TEMP_F'] = (value / 10.0) * 9/5 + 32
        # Precipitation is in tenths of mm
        elif element == 'PRCP':
            result.loc[index, 'PRECIP_MM'] = value / 10.0
            result.loc[index, 'PRECIP_IN'] = (value / 10.0) / 25.4
        # Snow depth is in mm
        elif element == 'SNWD':
            result.loc[index, 'SNOW_DEPTH_MM'] = value
            result.loc[index, 'SNOW_DEPTH_IN'] = value / 25.4
        # Snowfall is in mm
        elif element == 'SNOW':
            result.loc[index, 'SNOWFALL_MM'] = value
            result.loc[index, 'SNOWFALL_IN'] = value / 25.4
    
    return result
# %%
station_id = "USW00094846"  # Chicago O'Hare International Airport

# Target date
date_str = "4/4/2024"

print(f"Retrieving GHCN-Daily data for {station_id} (Chicago O'Hare) on {date_str}...")

# Get the raw data
raw_data = get_ghcnd_data(station_id, date_str)

if raw_data is not None:
    print("\nRaw GHCN-Daily data:")
    print(raw_data)
    
    # Parse and interpret the data
    parsed_data = parse_ghcnd_data(raw_data)
    
    if parsed_data is not None:
        print("\nInterpreted weather data:")
        print(parsed_data)
    else:
        print("Failed to parse the weather data.")
else:
    print("No data retrieved.")
# %%