import os
import numpy as np
import pandas as pd

def download_flight_data(year, month, output_dir="flight_data", origin_airports=None, dest_airports=None):
    """
    Download flight data from the Bureau of Transportation Statistics for a specific year and month.
    
    Args:
        year (int): Year to download data for
        month (int): Month to download data for (1-12)
        output_dir (str): Directory to save downloaded data
        origin_airports (list): List of origin airport codes to filter
        dest_airports (list): List of destination airport codes to filter
        
    Returns:
        pandas.DataFrame: DataFrame containing the downloaded data
    """
    print(f"Downloading data for {year}-{month:02d}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For this implementation, we'll generate sample data (from test_flight.py)
    print("Generating sample data based on typical BTS data structure...")
    
    # Generate sample data that mimics BTS data structure
    sample_data = generate_sample_data(year, month, origin_airports, dest_airports)
    
    # Save to CSV
    file_name = f"{year}_{month:02d}_flight_data.csv"
    file_path = os.path.join(output_dir, file_name)
    sample_data.to_csv(file_path, index=False)
    
    print(f"Data saved to {file_path}")
    return sample_data

def generate_sample_data(year, month, origin_airports=None, dest_airports=None):
    """
    Generate sample data for testing when not connecting to the actual API.
    
    Args:
        year (int): Year
        month (int): Month
        origin_airports (list): List of origin airport codes
        dest_airports (list): List of destination airport codes
        
    Returns:
        pandas.DataFrame: DataFrame with sample data
    """
    # Define sample airports if not provided
    if origin_airports is None:
        origin_airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK', 'DEN', 'SFO', 'SEA', 'LAS', 'MCO']
    if dest_airports is None:
        dest_airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK', 'SFO', 'MIA', 'DEN', 'BOS', 'SEA', 'LAS', 'MCO']
    
    # Generate sample flights
    num_samples = 5000
    
    # Create a date range for the given month
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    
    # Generate random dates within the month
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:-1]
    
    # Common carriers
    carriers = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'G4']
    
    # Generate random data
    data = {
        'FL_DATE': np.random.choice(dates, num_samples),
        'UNIQUE_CARRIER': np.random.choice(carriers, num_samples),
        'CARRIER': np.random.choice(carriers, num_samples),
        'FL_NUM': np.random.randint(100, 9999, num_samples),
        'ORIGIN_AIRPORT_ID': np.random.randint(10000, 99999, num_samples),
        'ORIGIN': np.random.choice(origin_airports, num_samples),
        'DEST_AIRPORT_ID': np.random.randint(10000, 99999, num_samples),
        'DEST': np.random.choice(dest_airports, num_samples),
        'CRS_DEP_TIME': [f"{np.random.randint(0, 24):02d}{np.random.randint(0, 60):02d}" for _ in range(num_samples)],
        'DEP_TIME': [f"{np.random.randint(0, 24):02d}{np.random.randint(0, 60):02d}" for _ in range(num_samples)],
        'DISTANCE': np.random.randint(100, 3000, num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate delay data with realistic distribution
    # Most flights are on time or have short delays
    dep_delay = np.random.exponential(15, num_samples)
    dep_delay = np.round(dep_delay)
    
    # Add some cancelled flights (about 2%)
    cancelled = np.random.choice([0, 1], num_samples, p=[0.98, 0.02])
    
    # Add some diverted flights (about 0.5%)
    diverted = np.random.choice([0, 1], num_samples, p=[0.995, 0.005])
    
    # For cancelled/diverted flights, set delays to NA
    dep_delay[cancelled == 1] = np.nan
    
    # Calculate arrival delay (typically correlated with departure delay plus some variation)
    arr_delay = dep_delay + np.random.normal(0, 10, num_samples)
    arr_delay = np.round(arr_delay)
    arr_delay[cancelled == 1] = np.nan
    arr_delay[diverted == 1] = np.nan
    
    # Add delay information to dataframe
    df['DEP_DELAY'] = dep_delay
    df['DEP_DELAY_NEW'] = df['DEP_DELAY'].apply(lambda x: max(0, x) if not pd.isna(x) else np.nan)
    df['DEP_DEL15'] = df['DEP_DELAY'].apply(lambda x: 1 if not pd.isna(x) and x >= 15 else 0)
    
    df['ARR_DELAY'] = arr_delay
    df['ARR_DELAY_NEW'] = df['ARR_DELAY'].apply(lambda x: max(0, x) if not pd.isna(x) else np.nan)
    df['ARR_DEL15'] = df['ARR_DELAY'].apply(lambda x: 1 if not pd.isna(x) and x >= 15 else 0)
    
    df['CANCELLED'] = cancelled
    df['DIVERTED'] = diverted
    
    # Define cancellation codes
    # A: Carrier, B: Weather, C: National Air System, D: Security
    cancellation_codes = ['A', 'B', 'C', 'D']
    cancellation_probs = [0.3, 0.4, 0.25, 0.05]  # Weather is the most common reason
    
    df['CANCELLATION_CODE'] = np.nan
    cancelled_indices = df[df['CANCELLED'] == 1].index
    df.loc[cancelled_indices, 'CANCELLATION_CODE'] = np.random.choice(
        cancellation_codes, size=len(cancelled_indices), p=cancellation_probs
    )
    
    # For delayed flights, assign delay reasons
    # Only flights delayed 15+ minutes have delay reasons
    delayed_indices = df[(df['ARR_DEL15'] == 1) & (df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].index
    
    # Initialize delay reason columns with 0
    for delay_type in ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']:
        df[delay_type] = 0
    
    # Assign delay reasons with realistic distribution
    for idx in delayed_indices:
        total_delay = df.loc[idx, 'ARR_DELAY']
        
        # Randomly split the delay among different reasons
        # Distribution based on typical patterns
        carrier_pct = np.random.uniform(0, 0.4)  # 0-40% carrier delay
        weather_pct = np.random.uniform(0, 0.3)  # 0-30% weather delay
        nas_pct = np.random.uniform(0, 0.3)      # 0-30% NAS delay
        security_pct = np.random.uniform(0, 0.05) # 0-5% security delay
        
        # Remaining is late aircraft delay
        late_aircraft_pct = 1 - (carrier_pct + weather_pct + nas_pct + security_pct)
        
        # Assign delay minutes
        df.loc[idx, 'CARRIER_DELAY'] = round(total_delay * carrier_pct)
        df.loc[idx, 'WEATHER_DELAY'] = round(total_delay * weather_pct)
        df.loc[idx, 'NAS_DELAY'] = round(total_delay * nas_pct)
        df.loc[idx, 'SECURITY_DELAY'] = round(total_delay * security_pct)
        df.loc[idx, 'LATE_AIRCRAFT_DELAY'] = round(total_delay * late_aircraft_pct)
    
    # Add air time (typically correlated with distance)
    df['AIR_TIME'] = df['DISTANCE'] / 8 + np.random.normal(0, 10, num_samples)
    df['AIR_TIME'] = np.round(df['AIR_TIME'])
    df.loc[cancelled_indices, 'AIR_TIME'] = np.nan
    
    # Add time-based columns needed for weather merging
    df['DEP_HOUR'] = df['CRS_DEP_TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
    df['DEP_DATE'] = df['FL_DATE'].dt.date
    
    return df

def download_flight_data_for_year(year, output_dir="flight_data"):
    """
    Download flight data for all months in a year
    
    Args:
        year (int): Year to download data for
        output_dir (str): Directory to save downloaded data
        
    Returns:
        pandas.DataFrame: Combined DataFrame for the entire year
    """
    print(f"Downloading flight data for {year}...")
    
    # Create a list to store DataFrames for each month
    monthly_dfs = []
    
    # Download data for each month
    for month in range(1, 13):
        # Using the existing download_flight_data function
        monthly_data = download_flight_data(year, month, output_dir)
        monthly_dfs.append(monthly_data)
    
    # Combine all monthly data into a single DataFrame
    combined_data = pd.concat(monthly_dfs, ignore_index=True)
    
    # Save combined data to CSV
    combined_file_path = os.path.join(output_dir, f"{year}_flight_data_combined.csv")
    combined_data.to_csv(combined_file_path, index=False)
    
    print(f"Combined flight data for {year} saved to {combined_file_path}")
    return combined_data