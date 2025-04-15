#%%
import pandas as pd
import requests
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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
    
    # The URL for the BTS data download form
    url = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    
    # Build request parameters
    params = {
        'Table_ID': '236',  # On-Time Performance table
        'Has_Group': '0',
        'Is_Zipped': '0',
        'Year': str(year),
        'Month': str(month),
        # Required fields for flight data and delay information
        'FIELDS': ','.join([
            'FL_DATE',               # Flight date
            'UNIQUE_CARRIER',        # Unique carrier code
            'CARRIER',               # Carrier code
            'FL_NUM',                # Flight number
            'ORIGIN_AIRPORT_ID',     # Origin airport ID
            'ORIGIN',                # Origin airport code
            'DEST_AIRPORT_ID',       # Destination airport ID
            'DEST',                  # Destination airport code
            'CRS_DEP_TIME',          # Scheduled departure time
            'DEP_TIME',              # Actual departure time
            'DEP_DELAY',             # Departure delay (minutes)
            'DEP_DELAY_NEW',         # Departure delay (minutes) > 0
            'DEP_DEL15',             # Departure delay > 15 minutes
            'CRS_ARR_TIME',          # Scheduled arrival time
            'ARR_TIME',              # Actual arrival time
            'ARR_DELAY',             # Arrival delay (minutes)
            'ARR_DELAY_NEW',         # Arrival delay (minutes) > 0
            'ARR_DEL15',             # Arrival delay > 15 minutes
            'CANCELLED',             # Cancelled flight indicator
            'CANCELLATION_CODE',     # Cancellation reason
            'DIVERTED',              # Diverted flight indicator
            'CARRIER_DELAY',         # Carrier delay (minutes)
            'WEATHER_DELAY',         # Weather delay (minutes)
            'NAS_DELAY',             # National Air System delay (minutes)
            'SECURITY_DELAY',        # Security delay (minutes)
            'LATE_AIRCRAFT_DELAY',   # Late aircraft delay (minutes)
            'AIR_TIME',              # Flight time (minutes)
            'DISTANCE'               # Distance between airports (miles)
        ])
    }
    
    # In a real implementation, this would involve session management and post requests
    # For now, we'll simulate a download with sample data
    
    print("Note: In a real implementation, this would download from the BTS website.")
    print("For now, generating sample data based on typical BTS data structure...")
    
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
    In a real implementation, this would be replaced with actual data from BTS.
    
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
        origin_airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK']
    if dest_airports is None:
        dest_airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK', 'SFO', 'MIA']
    
    # Generate 1000 sample flights
    num_samples = 1000
    
    # Create a date range for the given month
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    
    # Generate random dates within the month
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:-1]
    
    # Common carriers
    carriers = ['AA', 'DL', 'UA', 'WN', 'B6']
    
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
    
    return df

def load_flight_data(file_path):
    """
    Load flight data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the loaded data
    """
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'FL_DATE' in data.columns:
        data['FL_DATE'] = pd.to_datetime(data['FL_DATE'])
    
    return data

def get_basic_stats(data):
    """
    Get basic statistics about the flight data
    
    Args:
        data (pandas.DataFrame): Flight data
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    total_flights = len(data)
    cancelled_flights = data['CANCELLED'].sum()
    diverted_flights = data['DIVERTED'].sum()
    on_time_flights = sum((data['ARR_DEL15'] == 0) & (data['CANCELLED'] == 0) & (data['DIVERTED'] == 0))
    delayed_flights = sum(data['ARR_DEL15'] == 1)
    
    # Calculate percentages
    pct_cancelled = cancelled_flights / total_flights * 100
    pct_diverted = diverted_flights / total_flights * 100
    pct_on_time = on_time_flights / total_flights * 100
    pct_delayed = delayed_flights / total_flights * 100
    
    # Calculate average delay times
    avg_dep_delay = data['DEP_DELAY_NEW'].mean()
    avg_arr_delay = data['ARR_DELAY_NEW'].mean()
    
    return {
        "total_flights": total_flights,
        "cancelled_flights": cancelled_flights,
        "pct_cancelled": pct_cancelled,
        "diverted_flights": diverted_flights,
        "pct_diverted": pct_diverted,
        "on_time_flights": on_time_flights,
        "pct_on_time": pct_on_time,
        "delayed_flights": delayed_flights,
        "pct_delayed": pct_delayed,
        "avg_dep_delay": avg_dep_delay,
        "avg_arr_delay": avg_arr_delay
    }

def analyze_delay_reasons(data):
    """
    Analyze the reasons for flight delays
    
    Args:
        data (pandas.DataFrame): Flight data
        
    Returns:
        dict: Dictionary containing delay reason statistics
    """
    # Filter to only include delayed flights
    delayed_flights = data[data['ARR_DEL15'] == 1]
    
    # Sum up the minutes for each delay reason
    total_carrier_delay = delayed_flights['CARRIER_DELAY'].sum()
    total_weather_delay = delayed_flights['WEATHER_DELAY'].sum()
    total_nas_delay = delayed_flights['NAS_DELAY'].sum()
    total_security_delay = delayed_flights['SECURITY_DELAY'].sum()
    total_late_aircraft_delay = delayed_flights['LATE_AIRCRAFT_DELAY'].sum()
    
    # Calculate the total delay minutes
    total_delay_minutes = (
        total_carrier_delay + 
        total_weather_delay + 
        total_nas_delay + 
        total_security_delay + 
        total_late_aircraft_delay
    )
    
    # Calculate percentages
    pct_carrier = total_carrier_delay / total_delay_minutes * 100 if total_delay_minutes > 0 else 0
    pct_weather = total_weather_delay / total_delay_minutes * 100 if total_delay_minutes > 0 else 0
    pct_nas = total_nas_delay / total_delay_minutes * 100 if total_delay_minutes > 0 else 0
    pct_security = total_security_delay / total_delay_minutes * 100 if total_delay_minutes > 0 else 0
    pct_late_aircraft = total_late_aircraft_delay / total_delay_minutes * 100 if total_delay_minutes > 0 else 0
    
    return {
        "total_delay_minutes": total_delay_minutes,
        "carrier_delay_minutes": total_carrier_delay,
        "pct_carrier": pct_carrier,
        "weather_delay_minutes": total_weather_delay,
        "pct_weather": pct_weather,
        "nas_delay_minutes": total_nas_delay,
        "pct_nas": pct_nas,
        "security_delay_minutes": total_security_delay,
        "pct_security": pct_security,
        "late_aircraft_delay_minutes": total_late_aircraft_delay,
        "pct_late_aircraft": pct_late_aircraft
    }

def analyze_cancellation_reasons(data):
    """
    Analyze the reasons for flight cancellations
    
    Args:
        data (pandas.DataFrame): Flight data
        
    Returns:
        dict: Dictionary containing cancellation reason statistics
    """
    # Filter to only include cancelled flights
    cancelled_flights = data[data['CANCELLED'] == 1]
    
    # Count cancellations by reason
    cancellation_counts = cancelled_flights['CANCELLATION_CODE'].value_counts()
    
    # Define cancellation reason mappings
    reason_map = {
        'A': 'Carrier',
        'B': 'Weather',
        'C': 'National Air System',
        'D': 'Security'
    }
    
    # Create a dictionary with counts and percentages
    result = {"total_cancellations": len(cancelled_flights)}
    
    for code, reason in reason_map.items():
        count = cancellation_counts.get(code, 0)
        pct = count / len(cancelled_flights) * 100 if len(cancelled_flights) > 0 else 0
        result[f"{reason.lower()}_cancellations"] = count
        result[f"pct_{reason.lower()}"] = pct
    
    return result

def plot_delay_reasons(data, save_path=None):
    """
    Plot the breakdown of delay reasons
    
    Args:
        data (pandas.DataFrame): Flight data
        save_path (str): Path to save the plot image (optional)
        
    Returns:
        dict: Message indicating plot was created or saved
    """
    delay_reasons = analyze_delay_reasons(data)
    
    # Extract delay minutes and percentages
    labels = [
        'Carrier\n(Airline)',
        'Weather',
        'National Air\nSystem',
        'Security',
        'Late\nAircraft'
    ]
    
    values = [
        delay_reasons["carrier_delay_minutes"],
        delay_reasons["weather_delay_minutes"],
        delay_reasons["nas_delay_minutes"],
        delay_reasons["security_delay_minutes"],
        delay_reasons["late_aircraft_delay_minutes"]
    ]
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        shadow=True, 
        startangle=90
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Flight Delay Reasons (by Minutes)', fontsize=16)
    
    if save_path:
        plt.savefig(save_path)
        return {"message": f"Plot saved to {save_path}"}
    else:
        plt.show()
        return {"message": "Plot displayed"}

def plot_cancellation_reasons(data, save_path=None):
    """
    Plot the breakdown of cancellation reasons
    
    Args:
        data (pandas.DataFrame): Flight data
        save_path (str): Path to save the plot image (optional)
        
    Returns:
        dict: Message indicating plot was created or saved
    """
    cancellation_reasons = analyze_cancellation_reasons(data)
    
    # Extract cancellation counts
    labels = [
        'Carrier\n(Airline)',
        'Weather',
        'National Air\nSystem',
        'Security'
    ]
    
    values = [
        cancellation_reasons.get("carrier_cancellations", 0),
        cancellation_reasons.get("weather_cancellations", 0),
        cancellation_reasons.get("national_air_system_cancellations", 0),
        cancellation_reasons.get("security_cancellations", 0)
    ]
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        shadow=True, 
        startangle=90
    )
    plt.axis('equal')
    plt.title('Flight Cancellation Reasons', fontsize=16)
    
    if save_path:
        plt.savefig(save_path)
        return {"message": f"Plot saved to {save_path}"}
    else:
        plt.show()
        return {"message": "Plot displayed"}
# %%
data = download_flight_data(2023, 1)
    
# Get basic statistics
stats = get_basic_stats(data)
print("\nBasic Flight Statistics:")
print(f"Total Flights: {stats['total_flights']}")
print(f"On-time Flights: {stats['on_time_flights']} ({stats['pct_on_time']:.1f}%)")
print(f"Delayed Flights: {stats['delayed_flights']} ({stats['pct_delayed']:.1f}%)")
print(f"Cancelled Flights: {stats['cancelled_flights']} ({stats['pct_cancelled']:.1f}%)")
print(f"Diverted Flights: {stats['diverted_flights']} ({stats['pct_diverted']:.1f}%)")
print(f"Average Departure Delay: {stats['avg_dep_delay']:.1f} minutes")
print(f"Average Arrival Delay: {stats['avg_arr_delay']:.1f} minutes")
# %%
delay_reasons = analyze_delay_reasons(data)
print("\nDelay Reason Analysis:")
print(f"Total Delay Minutes: {delay_reasons['total_delay_minutes']:.0f}")
print(f"Carrier Delay: {delay_reasons['carrier_delay_minutes']:.0f} minutes ({delay_reasons['pct_carrier']:.1f}%)")
print(f"Weather Delay: {delay_reasons['weather_delay_minutes']:.0f} minutes ({delay_reasons['pct_weather']:.1f}%)")
print(f"National Air System Delay: {delay_reasons['nas_delay_minutes']:.0f} minutes ({delay_reasons['pct_nas']:.1f}%)")
print(f"Security Delay: {delay_reasons['security_delay_minutes']:.0f} minutes ({delay_reasons['pct_security']:.1f}%)")
print(f"Late Aircraft Delay: {delay_reasons['late_aircraft_delay_minutes']:.0f} minutes ({delay_reasons['pct_late_aircraft']:.1f}%)")
# %%
cancellation_reasons = analyze_cancellation_reasons(data)
print("\nCancellation Reason Analysis:")
print(f"Total Cancellations: {cancellation_reasons['total_cancellations']}")
print(f"Carrier Cancellations: {cancellation_reasons.get('carrier_cancellations', 0)} ({cancellation_reasons.get('pct_carrier', 0):.1f}%)")
print(f"Weather Cancellations: {cancellation_reasons.get('weather_cancellations', 0)} ({cancellation_reasons.get('pct_weather', 0):.1f}%)")
print(f"National Air System Cancellations: {cancellation_reasons.get('national_air_system_cancellations', 0)} ({cancellation_reasons.get('pct_national_air_system', 0):.1f}%)")
print(f"Security Cancellations: {cancellation_reasons.get('security_cancellations', 0)} ({cancellation_reasons.get('pct_security', 0):.1f}%)")
# %%
