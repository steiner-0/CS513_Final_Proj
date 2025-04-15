import os
import numpy as np
import pandas as pd

def generate_sample_weather_data(year, airport_code):
    """
    Generate sample weather data for an airport for a full year
    
    Args:
        year (int): Year
        airport_code (str): Airport code
        
    Returns:
        pandas.DataFrame: DataFrame with sample weather data
    """
    # Create a date range for the entire year
    date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
    
    # Create a DataFrame with the date range as index
    weather_df = pd.DataFrame(index=date_range)
    
    # Add weather variables with seasonal patterns
    # Temperature average
    base_temp = 15  # Base temperature in Celsius
    amplitude = 15  # Amplitude of seasonal variation
    
    # Create seasonal pattern with some randomness
    days = np.arange(len(date_range))
    seasonal_temp = base_temp + amplitude * np.sin(2 * np.pi * days / 365)
    
    # Add random variation
    weather_df['tavg'] = seasonal_temp + np.random.normal(0, 3, len(date_range))
    weather_df['tmin'] = weather_df['tavg'] - np.random.uniform(3, 8, len(date_range))
    weather_df['tmax'] = weather_df['tavg'] + np.random.uniform(3, 8, len(date_range))
    
    # Precipitation (prcp)
    # More precipitation in winter and early spring months
    prcp_base = np.zeros(len(date_range))
    winter_mask = (weather_df.index.month >= 11) | (weather_df.index.month <= 3)
    summer_mask = (weather_df.index.month >= 6) & (weather_df.index.month <= 8)
    
    prcp_base[winter_mask] = 5  # Higher base precipitation in winter
    prcp_base[summer_mask] = 2  # Lower base precipitation in summer
    
    # Add random precipitation with many zeros (dry days)
    prcp_random = np.random.exponential(1, len(date_range)) * prcp_base
    prcp_zeros = np.random.choice([0, 1], len(date_range), p=[0.7, 0.3])  # 70% chance of dry day
    weather_df['prcp'] = prcp_random * prcp_zeros
    
    # Snow (snow) - only in cold months
    weather_df['snow'] = 0.0
    snow_months = (weather_df.index.month >= 11) | (weather_df.index.month <= 3)
    cold_days = weather_df['tavg'] < 2  # Snow only when cold enough
    snow_days = snow_months & cold_days
    
    if sum(snow_days) > 0:
        weather_df.loc[snow_days, 'snow'] = np.random.exponential(1, sum(snow_days)) * 5
    
    # Wind speed (wspd)
    weather_df['wspd'] = np.random.gamma(2, 2, len(date_range))
    
    # Pressure (pres)
    base_pressure = 1013.25  # Standard sea level pressure in hPa
    weather_df['pres'] = base_pressure + np.random.normal(0, 5, len(date_range))
    
    # Cloud cover (coco) - percentage
    weather_df['coco'] = np.random.beta(2, 2, len(date_range)) * 100
    
    # Add some missing values to simulate real data
    for col in weather_df.columns:
        # Randomly set 5% of values to NaN
        mask = np.random.choice([False, True], len(weather_df), p=[0.95, 0.05])
        weather_df.loc[mask, col] = np.nan
    
    # Convert index to DatetimeIndex
    weather_df.index = pd.DatetimeIndex(weather_df.index)
    weather_df.index.name = 'date'
    
    # Add airport code and day of week
    weather_df['airport'] = airport_code
    weather_df['weekday'] = weather_df.index.dayofweek
    
    return weather_df

def download_weather_data(year, airports, output_dir="weather_data"):
    """
    Download weather data for each airport in the list for the specified year
    
    Args:
        year (int): Year to download data for
        airports (list): List of airport codes to get weather data for
        output_dir (str): Directory to save downloaded data
        
    Returns:
        dict: Dictionary mapping airport codes to their weather DataFrames
    """
    print(f"Downloading weather data for {year}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dictionary to store weather data by airport
    airport_weather = {}
    
    # For each airport, generate sample weather data
    for airport in airports:
        print(f"Getting weather data for {airport}...")
        
        try:
            # Generate sample weather data
            weather_data = generate_sample_weather_data(year, airport)
            
            # Save to CSV
            file_name = f"{year}_{airport}_weather_data.csv"
            file_path = os.path.join(output_dir, file_name)
            weather_data.to_csv(file_path, index=True)
            
            print(f"Weather data for {airport} saved to {file_path}")
            
            # Store in dictionary
            airport_weather[airport] = weather_data
            
        except Exception as e:
            print(f"Error getting weather data for {airport}: {e}")
    
    return airport_weather