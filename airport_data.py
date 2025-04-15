import os
import numpy as np
import pandas as pd

def create_airport_metadata(airports, output_dir="airport_data"):
    """
    Create metadata for airports
    
    Args:
        airports (list): List of airport codes
        output_dir (str): Directory to save the data
        
    Returns:
        pandas.DataFrame: DataFrame with airport metadata
    """
    print("Creating airport metadata...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sample airport data
    data = []
    base_lat, base_lon = 40.0, -100.0
    
    for i, code in enumerate(airports):
        # Sample data with variations
        lat = base_lat + (i * 0.5) + np.random.normal(0, 0.1)
        lon = base_lon + (i * 0.5) + np.random.normal(0, 0.1)
        
        # Create name based on code
        if len(code) == 3:
            name = f"{code} International Airport"
        else:
            name = f"{code} Regional Airport"
        
        # Region (simplistic assignment for demonstration)
        if i % 5 == 0:
            region = "Northeast"
        elif i % 5 == 1:
            region = "Southeast"
        elif i % 5 == 2:
            region = "Midwest"
        elif i % 5 == 3:
            region = "Southwest"
        else:
            region = "West"
        
        # City (simplistic assignment for demonstration)
        city = f"City{i+1}"
        
        data.append({
            'airport_code': code,
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'elevation_ft': np.random.randint(0, 7000),
            'city': city,
            'state': f"State{i%50+1}",
            'region': region,
            'is_major': np.random.choice([0, 1], p=[0.7, 0.3])
        })
    
    # Create DataFrame
    airport_df = pd.DataFrame(data)
    
    # Add some missing values to simulate real data
    for col in ['elevation_ft', 'state']:
        # Randomly set 5% of values to NaN
        mask = np.random.choice([False, True], len(airport_df), p=[0.95, 0.05])
        airport_df.loc[mask, col] = np.nan
    
    # Save to CSV
    file_path = os.path.join(output_dir, "airport_metadata.csv")
    airport_df.to_csv(file_path, index=False)
    
    print(f"Airport metadata saved to {file_path}")
    return airport_df