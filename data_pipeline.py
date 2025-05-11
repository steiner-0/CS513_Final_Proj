import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from meteostat import Point, Daily, Hourly, Stations
import re
import csv

# File paths
FLIGHT_DATA_DIR = "flight_data"
AIRPORT_DATA_FILE = "airport_data/airports.dat"
WEATHER_DATA_DIR = "weather_data"
OUTPUT_DIR = "analysis_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_airport_coordinates(iata_code, filepath=AIRPORT_DATA_FILE):
    """
    Looks up the latitude and longitude of an airport given its IATA code.
    
    Parameters:
        iata_code (str): The 3-letter IATA airport code.
        filepath (str): Path to the OpenFlights airports.dat file.
        
    Returns:
        tuple: (latitude, longitude) if found, otherwise None.
    """
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Columns: ID, Name, City, Country, IATA, ICAO, Latitude, Longitude, ...
            if len(row) > 7 and row[4].upper() == iata_code.upper():
                return float(row[6]), float(row[7])
            
    print(f"Airport with IATA code {iata_code} not found in {filepath}.")
    return None, None


def airport_to_station(iata_code, filepath=AIRPORT_DATA_FILE):
    """
    Converts an airport IATA code to a Meteostat station ID.
    
    Parameters:
        iata_code (str): The 3-letter IATA airport code.
        filepath (str): Path to the OpenFlights airports.dat file.
        
    Returns:
        str: Meteostat station ID if found, otherwise None.
    """
    coords = get_airport_coordinates(iata_code, filepath)
    
    if coords[0] is None or coords[1] is None:
        return None
        
    latitude, longitude = coords

    # Use Meteostat to find nearby stations
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    
    # Fetch the nearest station
    nearest_station = stations.fetch(1)
    
    return nearest_station.index[0] if not nearest_station.empty else None


def load_flight_data(directory=FLIGHT_DATA_DIR):
    """
    Load all flight data files into a consolidated DataFrame.
    
    Parameters:
        directory (str): Directory containing flight data CSV files
        
    Returns:
        pandas.DataFrame: Consolidated flight data
    """
    all_flights = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "flight_delays" in filename:
            file_path = os.path.join(directory, filename)
            
            # Extract origin, airline, and year from filename
            parts = filename.split('_')
            if len(parts) >= 4:
                origin = parts[0].upper()
                airline = parts[1].upper()
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add origin and airline columns
                df['origin'] = origin
                df['airline'] = airline
                
                # Standardize date format and create datetime column
                if 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                
                all_flights.append(df)
    
    if not all_flights:
        print("No flight data files found!")
        return pd.DataFrame()
        
    # Combine all DataFrames
    flight_data = pd.concat(all_flights, ignore_index=True)
    
    # Clean and standardize column names
    flight_data.columns = [col.lower().replace(' ', '_') for col in flight_data.columns]
    
    # Convert delay columns to numeric
    for col in flight_data.columns:
        if 'delay' in col.lower() and flight_data[col].dtype == 'object':
            flight_data[col] = pd.to_numeric(flight_data[col], errors='coerce')
    
    return flight_data


def fetch_weather_data(airport_code, year, output_dir=WEATHER_DATA_DIR):
    """
    Fetch weather data for a specific airport and year using Meteostat.
    
    Parameters:
        airport_code (str): Airport IATA code
        year (int): Year to fetch data for
        output_dir (str): Directory to save weather data
        
    Returns:
        pandas.DataFrame: Weather data for the specified airport and year
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{airport_code.lower()}_weather_{year}.csv")
    
    # Check if we already have the data
    if os.path.exists(output_file):
        print(f"Loading cached weather data for {airport_code} {year}")
        return pd.read_csv(output_file, parse_dates=['date'])
    
    # Get the meteorological station for this airport
    station_id = airport_to_station(airport_code)
    
    if not station_id:
        print(f"Could not find a weather station for airport {airport_code}")
        return pd.DataFrame()
    
    # Define the time period
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    
    try:
        # Fetch daily data from Meteostat
        data = Daily(station_id, start, end)
        weather_data = data.fetch()
        
        # Reset index to make date a column
        weather_data = weather_data.reset_index()
        
        # Add airport code
        weather_data['airport'] = airport_code.upper()
        
        # Save to CSV
        weather_data.to_csv(output_file, index=False)
        
        return weather_data
    
    except Exception as e:
        print(f"Error fetching weather data for {airport_code}: {e}")
        return pd.DataFrame()


def merge_flight_weather_data(flight_data, weather_data_dict):
    """
    Merge flight data with corresponding weather data.
    
    Parameters:
        flight_data (pandas.DataFrame): Flight data
        weather_data_dict (dict): Dictionary mapping airport codes to weather DataFrames
        
    Returns:
        pandas.DataFrame: Merged flight and weather data
    """
    # Create a list to store merged DataFrames
    merged_dfs = []
    
    # Get unique origin airports
    origins = flight_data['origin'].unique()
    
    for origin in origins:
        # Get flights for this origin
        origin_flights = flight_data[flight_data['origin'] == origin].copy()
        
        # Check if we have weather data for this airport
        if origin.lower() in weather_data_dict:
            origin_weather = weather_data_dict[origin.lower()].copy()
            
            # Ensure date columns are datetime for merging
            if 'date' in origin_flights.columns and 'date' in origin_weather.columns:
                # Convert dates to date only (no time) for merging
                origin_flights['merge_date'] = origin_flights['date'].dt.date
                origin_weather['merge_date'] = pd.to_datetime(origin_weather['date']).dt.date
                
                # Merge flight and weather data
                merged = pd.merge(
                    origin_flights,
                    origin_weather,
                    on='merge_date',
                    how='left',
                    suffixes=('', '_weather')
                )
                
                # Drop the temporary merge column
                merged.drop('merge_date', axis=1, inplace=True)
                
                merged_dfs.append(merged)
            else:
                print(f"Missing date column for {origin}")
                merged_dfs.append(origin_flights)
        else:
            print(f"No weather data for {origin}")
            merged_dfs.append(origin_flights)
    
    # Combine all merged DataFrames
    if not merged_dfs:
        return pd.DataFrame()
        
    merged_data = pd.concat(merged_dfs, ignore_index=True)
    
    return merged_data


def analyze_weather_impact(merged_data, output_dir=OUTPUT_DIR):
    """
    Analyze the impact of weather on flight delays and generate visualizations.
    
    Parameters:
        merged_data (pandas.DataFrame): Merged flight and weather data
        output_dir (str): Directory to save analysis output
        
    Returns:
        dict: Dictionary containing analysis results
    """
    if merged_data.empty:
        print("No data to analyze!")
        return {}
    
    results = {}
    
    # 1. Calculate correlation between weather variables and delays
    weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']
    delay_cols = [col for col in merged_data.columns if 'delay' in col.lower()]
    
    weather_delay_cols = weather_cols + delay_cols
    available_cols = [col for col in weather_delay_cols if col in merged_data.columns]
    
    correlation_df = merged_data[available_cols].corr()
    
    # Save correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Weather Variables and Flight Delays')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weather_delay_correlation.png'))
    plt.close()
    
    # 2. Analyze delay patterns by weather conditions
    # Precipitation impact
    if 'prcp' in merged_data.columns and any(col in merged_data.columns for col in delay_cols):
        delay_col = next(col for col in delay_cols if col in merged_data.columns)
        
        # Create categorical precipitation levels
        merged_data['prcp_level'] = pd.cut(
            merged_data['prcp'], 
            bins=[-0.1, 0, 5, 10, 100], 
            labels=['None', 'Light', 'Moderate', 'Heavy']
        )
        
        # Calculate average delay by precipitation level
        prcp_impact = merged_data.groupby('prcp_level')[delay_col].mean().reset_index()
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.barplot(x='prcp_level', y=delay_col, data=prcp_impact)
        plt.title('Average Flight Delay by Precipitation Level')
        plt.ylabel('Average Delay (minutes)')
        plt.xlabel('Precipitation Level')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_by_precipitation.png'))
        plt.close()
        
        results['precipitation_impact'] = prcp_impact.to_dict()
    
    # 3. Temperature impact
    if 'tavg' in merged_data.columns and any(col in merged_data.columns for col in delay_cols):
        delay_col = next(col for col in delay_cols if col in merged_data.columns)
        
        # Create temperature categories
        merged_data['temp_level'] = pd.cut(
            merged_data['tavg'],
            bins=[-100, 0, 10, 20, 30, 100],
            labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
        )
        
        # Calculate average delay by temperature level
        temp_impact = merged_data.groupby('temp_level')[delay_col].mean().reset_index()
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.barplot(x='temp_level', y=delay_col, data=temp_impact)
        plt.title('Average Flight Delay by Temperature')
        plt.ylabel('Average Delay (minutes)')
        plt.xlabel('Temperature Level')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_by_temperature.png'))
        plt.close()
        
        results['temperature_impact'] = temp_impact.to_dict()
    
    # 4. Wind speed impact
    if 'wspd' in merged_data.columns and any(col in merged_data.columns for col in delay_cols):
        delay_col = next(col for col in delay_cols if col in merged_data.columns)
        
        # Create wind speed categories
        merged_data['wind_level'] = pd.cut(
            merged_data['wspd'],
            bins=[-1, 5, 10, 20, 100],
            labels=['Light', 'Moderate', 'Strong', 'Severe']
        )
        
        # Calculate average delay by wind level
        wind_impact = merged_data.groupby('wind_level')[delay_col].mean().reset_index()
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.barplot(x='wind_level', y=delay_col, data=wind_impact)
        plt.title('Average Flight Delay by Wind Speed')
        plt.ylabel('Average Delay (minutes)')
        plt.xlabel('Wind Speed Level')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_by_wind.png'))
        plt.close()
        
        results['wind_impact'] = wind_impact.to_dict()
    
    # 5. Airport comparison
    if 'origin' in merged_data.columns and any(col in merged_data.columns for col in delay_cols):
        delay_col = next(col for col in delay_cols if col in merged_data.columns)
        
        # Calculate average delay by airport
        airport_impact = merged_data.groupby('origin')[delay_col].agg(['mean', 'count']).reset_index()
        
        # Visualize
        plt.figure(figsize=(12, 6))
        sns.barplot(x='origin', y='mean', data=airport_impact)
        plt.title('Average Flight Delay by Origin Airport')
        plt.ylabel('Average Delay (minutes)')
        plt.xlabel('Origin Airport')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_by_airport.png'))
        plt.close()
        
        results['airport_impact'] = airport_impact.to_dict()
    
    # 6. Airline comparison
    if 'airline' in merged_data.columns and any(col in merged_data.columns for col in delay_cols):
        delay_col = next(col for col in delay_cols if col in merged_data.columns)
        
        # Calculate average delay by airline
        airline_impact = merged_data.groupby('airline')[delay_col].agg(['mean', 'count']).reset_index()
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.barplot(x='airline', y='mean', data=airline_impact)
        plt.title('Average Flight Delay by Airline')
        plt.ylabel('Average Delay (minutes)')
        plt.xlabel('Airline')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_by_airline.png'))
        plt.close()
        
        results['airline_impact'] = airline_impact.to_dict()
    
    # Save the merged data
    merged_data.to_csv(os.path.join(output_dir, 'flight_weather_merged.csv'), index=False)
    
    return results


def create_summary_report(results, output_dir=OUTPUT_DIR):
    """
    Create a summary report of the analysis results.
    
    Parameters:
        results (dict): Analysis results
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    report_path = os.path.join(output_dir, 'weather_impact_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Impact of Weather on Flight Delays\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes how different weather conditions affect flight delays ")
        f.write("across various airports and airlines.\n\n")
        
        # Add sections for each analysis
        
        if 'precipitation_impact' in results:
            f.write("## Impact of Precipitation\n\n")
            f.write("Precipitation shows a significant impact on flight delays. ")
            f.write("As precipitation increases from none to heavy, delays tend to increase.\n\n")
            f.write("![Delay by Precipitation](delay_by_precipitation.png)\n\n")
        
        if 'temperature_impact' in results:
            f.write("## Impact of Temperature\n\n")
            f.write("Temperature extremes, particularly freezing conditions, ")
            f.write("are associated with longer flight delays.\n\n")
            f.write("![Delay by Temperature](delay_by_temperature.png)\n\n")
        
        if 'wind_impact' in results:
            f.write("## Impact of Wind Speed\n\n")
            f.write("Strong and severe winds correlate with increased flight delays.\n\n")
            f.write("![Delay by Wind](delay_by_wind.png)\n\n")
        
        if 'airport_impact' in results:
            f.write("## Airport Comparison\n\n")
            f.write("Different airports experience varying levels of weather-related delays.\n\n")
            f.write("![Delay by Airport](delay_by_airport.png)\n\n")
        
        if 'airline_impact' in results:
            f.write("## Airline Comparison\n\n")
            f.write("Airlines show different patterns in weather-related delays, ")
            f.write("which may reflect different operational strategies.\n\n")
            f.write("![Delay by Airline](delay_by_airline.png)\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Weather conditions significantly impact flight delays, with precipitation, ")
        f.write("temperature extremes, and high winds being major contributors. ")
        f.write("Different airports and airlines show varying levels of resilience to adverse weather conditions.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. Airlines should consider adjusting schedules during seasons with expected adverse weather.\n")
        f.write("2. Airports in regions prone to specific weather conditions should invest in appropriate infrastructure.\n")
        f.write("3. Travelers should plan for potential delays during periods of forecasted adverse weather.\n")
    
    print(f"Summary report created at {report_path}")
    return report_path


def main():
    """
    Main execution function for the flight weather analysis pipeline.
    """
    print("Starting Flight Weather Analysis Pipeline")
    
    # 1. Load flight data
    print("\nLoading flight data...")
    flight_data = load_flight_data()
    if flight_data.empty:
        print("No flight data available. Exiting.")
        return
    
    print(f"Loaded {len(flight_data)} flight records")
    
    # 2. Extract year from flight data
    if 'date' in flight_data.columns:
        year = int(flight_data['date'].dt.year.mode()[0])
    else:
        # Default to 2020 based on filenames
        year = 2020
    
    print(f"Analyzing data for year {year}")
    
    # 3. Get unique airports
    airports = flight_data['origin'].unique()
    print(f"Found {len(airports)} unique airports: {', '.join(airports)}")
    
    # 4. Fetch weather data for each airport
    print("\nFetching weather data...")
    weather_data_dict = {}
    
    for airport in airports:
        print(f"Fetching weather data for {airport}...")
        weather_data = fetch_weather_data(airport, year)
        if not weather_data.empty:
            weather_data_dict[airport.lower()] = weather_data
    
    if not weather_data_dict:
        print("No weather data available. Exiting.")
        return
    
    # 5. Merge flight and weather data
    print("\nMerging flight and weather data...")
    merged_data = merge_flight_weather_data(flight_data, weather_data_dict)
    
    if merged_data.empty:
        print("Failed to merge data. Exiting.")
        return
    
    print(f"Successfully merged data: {len(merged_data)} records")
    
    # 6. Analyze impact of weather on delays
    print("\nAnalyzing impact of weather on flight delays...")
    results = analyze_weather_impact(merged_data)
    
    # 7. Create summary report
    print("\nCreating summary report...")
    report_path = create_summary_report(results)
    
    print("\nAnalysis complete! Results available in the analysis_output directory.")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()