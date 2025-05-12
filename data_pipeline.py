import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Hourly, Stations
from tqdm import tqdm

# File paths
FLIGHT_DATA_DIR = "flight_data"
AIRPORT_DATA_FILE = "airport_data/airports.dat"
WEATHER_DATA_DIR = "weather_data"
OUTPUT_DIR = "output"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEATHER_DATA_DIR, exist_ok=True)

def get_airport_coordinates(iata_code, filepath=AIRPORT_DATA_FILE):
    """Get latitude and longitude for an airport from its IATA code"""
    try:
        # Open the airports.dat file and search for the airport
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                # Check if this line has the IATA code we're looking for
                if len(parts) > 4 and parts[4].strip('"') == iata_code:
                    # Return latitude and longitude
                    return float(parts[6]), float(parts[7])
        print(f"Airport {iata_code} not found in database")
        return None, None
    except Exception as e:
        print(f"Error finding coordinates for {iata_code}: {e}")
        return None, None

def get_nearest_weather_station(lat, lon):
    """Find the nearest weather station to the given coordinates"""
    if lat is None or lon is None:
        return None
    
    try:
        # Find nearby weather stations using Meteostat
        stations = Stations()
        stations = stations.nearby(lat, lon)
        nearest = stations.fetch(1)
        
        if nearest.empty:
            return None
        
        return nearest.index[0]  # Return the station ID
    except Exception as e:
        print(f"Error finding weather station: {e}")
        return None

def load_flight_data():
    """Load all flight CSV files from the flight_data directory"""
    print("Loading flight data...")
    all_data = []
    
    # Get all CSV files in the directory
    try:
        csv_files = [f for f in os.listdir(FLIGHT_DATA_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {FLIGHT_DATA_DIR}")
            return pd.DataFrame()
            
        # Process each CSV file with progress bar
        for file in tqdm(csv_files, desc="Loading flight data files"):
            file_path = os.path.join(FLIGHT_DATA_DIR, file)
            
            # Load only the columns we need
            cols_to_use = ['YEAR', 'FL_DATE', 'ORIGIN', 'DEST', 
                          'CRS_DEP_TIME', 'DEP_TIME', 
                          'CRS_ARR_TIME', 'ARR_TIME', 'WEATHER_DELAY']
            
            df = pd.read_csv(file_path, usecols=lambda c: c in cols_to_use)
            all_data.append(df)
            tqdm.write(f"Loaded {len(df)} records from {file}")
        
        # Combine all data
        if not all_data:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert date column to datetime
        combined_df['FL_DATE'] = pd.to_datetime(combined_df['FL_DATE'])
        
        # Create departure datetime for merging with hourly weather
        combined_df['DEP_HOUR'] = combined_df['CRS_DEP_TIME'].apply(
            lambda x: int(str(int(x)).zfill(4)[:2]) if pd.notna(x) else np.nan
        )
        combined_df['DEP_MINUTE'] = combined_df['CRS_DEP_TIME'].apply(
            lambda x: int(str(int(x)).zfill(4)[2:]) if pd.notna(x) else np.nan
        )
        
        # Create full departure datetime
        combined_df['DEP_DATETIME'] = combined_df.apply(
            lambda row: row['FL_DATE'] + timedelta(hours=int(row['DEP_HOUR']), minutes=int(row['DEP_MINUTE']))
            if pd.notna(row['DEP_HOUR']) and pd.notna(row['DEP_MINUTE']) else pd.NaT,
            axis=1
        )
        
        # Fill NaN values in WEATHER_DELAY with 0 (assuming no delay)
        combined_df['WEATHER_DELAY'] = combined_df['WEATHER_DELAY'].fillna(0)
        
        print(f"Successfully loaded {len(combined_df)} flight records")
        return combined_df
        
    except Exception as e:
        print(f"Error loading flight data: {e}")
        return pd.DataFrame()

def get_hourly_weather(airport_code, year):
    """Get hourly weather data for an airport in a specific year"""
    # Check if we already have cached data
    cache_file = os.path.join(WEATHER_DATA_DIR, f"{airport_code}_weather_{year}.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading cached weather data for {airport_code}")
        return pd.read_csv(cache_file, parse_dates=['time'])
    
    # Get airport coordinates
    lat, lon = get_airport_coordinates(airport_code)
    if lat is None or lon is None:
        print(f"Could not find coordinates for {airport_code}")
        return pd.DataFrame()
    
    # Find nearest weather station
    station_id = get_nearest_weather_station(lat, lon)
    if station_id is None:
        print(f"No weather station found near {airport_code}")
        return pd.DataFrame()
    
    # Define time period
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59)
    
    try:
        # Fetch hourly weather data
        print(f"Fetching weather data for {airport_code}...")
        data = Hourly(station_id, start, end)
        weather_df = data.fetch()
        
        # Reset index to make time a column
        weather_df = weather_df.reset_index()
        
        # Add airport code
        weather_df['airport'] = airport_code
        
        # Save to cache
        weather_df.to_csv(cache_file, index=False)
        
        return weather_df
    except Exception as e:
        print(f"Error fetching weather data for {airport_code}: {e}")
        return pd.DataFrame()

def merge_flight_and_weather(flight_data):
    """Merge flight data with weather data from origin and destination airports"""
    print("Merging flight and weather data...")
    
    if flight_data.empty:
        return pd.DataFrame()
    
    # Get unique airports and most common year
    airports = list(set(flight_data['ORIGIN'].unique()) | set(flight_data['DEST'].unique()))
    year = int(flight_data['YEAR'].mode()[0]) if 'YEAR' in flight_data.columns else \
           int(flight_data['FL_DATE'].dt.year.mode()[0])
    
    print(f"Processing {len(airports)} airports for year {year}")
    
    # Dictionary to store weather data by airport
    weather_data = {}
    
    # Fetch weather data for each airport with progress bar
    for airport in tqdm(airports, desc="Fetching weather data for airports"):
        weather_df = get_hourly_weather(airport, year)
        if not weather_df.empty:
            weather_data[airport] = weather_df
            tqdm.write(f"Successfully processed weather data for {airport}")
    
    # Create a copy of flight data for merging
    merged_df = flight_data.copy()
    
    # Add columns for origin and destination weather
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
    
    # Initialize weather columns with NaN
    for col in weather_cols:
        merged_df[f'origin_{col}'] = np.nan
        merged_df[f'dest_{col}'] = np.nan
    
    # Process each flight
    print("Merging weather data with flights...")
    
    # Group by origin and departure datetime to process more efficiently
    origins = merged_df['ORIGIN'].unique()
    for origin in tqdm(origins, desc="Merging data by origin airport"):
        if origin in weather_data:
            origin_group = merged_df[merged_df['ORIGIN'] == origin]
            origin_weather = weather_data[origin]
            
            # Convert to datetime and round to hour for merging
            origin_group['dep_hour'] = origin_group['DEP_DATETIME'].dt.floor('H')
            
            # Merge with weather data
            merged_origin = pd.merge(
                origin_group,
                origin_weather,
                left_on='dep_hour',
                right_on='time',
                how='left'
            )
            
            # Update original dataframe with weather data
            for col in weather_cols:
                if col in origin_weather.columns:
                    merged_df.loc[merged_origin.index, f'origin_{col}'] = merged_origin[col].values
            
            tqdm.write(f"Merged {len(origin_group)} flights for {origin}")
    
    # Clean up temporary columns
    if 'dep_hour' in merged_df.columns:
        merged_df.drop('dep_hour', axis=1, inplace=True)
    
    # Save merged data
    output_file = os.path.join(OUTPUT_DIR, 'flight_weather_merged.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")
    
    return merged_df

def main():
    """Run the data pipeline to merge flight and weather data"""
    print("Starting Flight Weather Data Pipeline")
    
    # Create a master progress bar for the entire pipeline
    pipeline_steps = tqdm(total=2, desc="Overall pipeline progress")
    
    try:
        # 1. Load flight data
        flight_data = load_flight_data()
        if flight_data.empty:
            print("No flight data available. Exiting.")
            return
        pipeline_steps.update(1)
        
        # 2. Merge flight data with weather data
        merged_data = merge_flight_and_weather(flight_data)
        if merged_data.empty:
            print("Failed to merge data. Exiting.")
            return
        pipeline_steps.update(1)
        
        print("\n✅ Data processing complete!")
        print(f"📊 Merged data saved to: {os.path.join(OUTPUT_DIR, 'flight_weather_merged.csv')}")
        print(f"📈 Total records processed: {len(merged_data)}")
        print(f"🌤️ Weather data cached in: {WEATHER_DATA_DIR}")
    
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        raise
    finally:
        # Close all progress bars
        pipeline_steps.close()

if __name__ == "__main__":
    main()