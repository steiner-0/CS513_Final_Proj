import os
import numpy as np
import pandas as pd

def clean_flight_data(data):
    """
    Clean the flight data
    
    Args:
        data (pandas.DataFrame): Flight data to clean
        
    Returns:
        pandas.DataFrame: Cleaned flight data
    """
    print("Cleaning flight data...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Convert date column to datetime if it's not already
    if 'FL_DATE' in df.columns and not pd.api.types.is_datetime64_dtype(df['FL_DATE']):
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # Handle missing values
    # For numeric columns, fill with the median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Skip certain columns where NaN might be meaningful (like delay for cancelled flights)
        if col not in ['DEP_DELAY', 'ARR_DELAY', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW',
                       'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with the most common value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Skip certain columns where NaN might be meaningful
        if col not in ['CANCELLATION_CODE']:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle outliers in delay columns
    delay_cols = ['DEP_DELAY', 'ARR_DELAY', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW']
    for col in delay_cols:
        if col in df.columns:
            # Calculate Q1, Q3, and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df[col] = df[col].apply(lambda x: 
                                   upper_bound if x > upper_bound and not pd.isna(x) 
                                   else (lower_bound if x < lower_bound and not pd.isna(x) else x))
    
    # Check for and remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Add computed columns for analysis
    # Add day of week
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    
    # Add month
    df['MONTH'] = df['FL_DATE'].dt.month
    
    # Parse departure time into hours and minutes for better merging with weather data
    if 'CRS_DEP_TIME' in df.columns:
        df['DEP_HOUR'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
        df['DEP_MINUTE'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4).str[2:].astype(int)
    
    # Create a date column without time for easier merging
    df['DEP_DATE'] = df['FL_DATE'].dt.date
    
    print(f"Flight data cleaned. Shape after cleaning: {df.shape}")
    return df
def clean_weather_data(data_dict):
    """
    Clean the weather data for each airport
    
    Args:
        data_dict (dict): Dictionary mapping airport codes to weather DataFrames
        
    Returns:
        dict: Dictionary mapping airport codes to cleaned weather DataFrames
    """
    print("Cleaning weather data...")
    
    cleaned_data = {}
    
    for airport, df in data_dict.items():
        print(f"Cleaning weather data for {airport}...")
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values
        # For temperature columns, use linear interpolation
        temp_cols = ['tavg', 'tmin', 'tmax', 'temp']
        for col in temp_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
        
        # For other numeric columns, use forward fill then backward fill
        numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in temp_cols and col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Handle outliers
        # For each numeric column, cap outliers
        for col in numeric_cols:
            if col in cleaned_df.columns:
                # Calculate Q1, Q3, and IQR
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                cleaned_df[col] = cleaned_df[col].apply(lambda x: 
                                       upper_bound if x > upper_bound and not pd.isna(x) 
                                       else (lower_bound if x < lower_bound and not pd.isna(x) else x))
        
        # Fill any remaining missing values with column medians
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
        
        # Store cleaned DataFrame in dictionary
        cleaned_data[airport] = cleaned_df
    
    print("Weather data cleaned.")
    return cleaned_data

def clean_airport_metadata(data):
    """
    Clean the airport metadata
    
    Args:
        data (pandas.DataFrame): Airport metadata to clean
        
    Returns:
        pandas.DataFrame: Cleaned airport metadata
    """
    print("Cleaning airport metadata...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    # For numeric columns, fill with the median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with the most common value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates from metadata
    df.drop_duplicates(inplace=True)
    
    # Convert coordinate columns to appropriate type
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    print(f"Airport metadata cleaned. Shape after cleaning: {df.shape}")
    return df

def merge_flight_and_weather(flight_data, weather_data_dict, airport_data):
    """
    Merge flight data with weather data based on origin airport and date
    
    Args:
        flight_data (pandas.DataFrame): Cleaned flight data
        weather_data_dict (dict): Dictionary of cleaned weather data by airport
        airport_data (pandas.DataFrame): Cleaned airport metadata
        
    Returns:
        pandas.DataFrame: Merged dataset
    """
    print("Merging flight and weather data...")
    
    # Create a list to store DataFrames for each airport
    merged_dfs = []
    
    # Process each origin airport
    unique_origins = flight_data['ORIGIN'].unique()
    
    for origin in unique_origins:
        # Get flights from this origin
        origin_flights = flight_data[flight_data['ORIGIN'] == origin].copy()
        
        # Check if we have weather data for this airport
        if origin in weather_data_dict:
            origin_weather = weather_data_dict[origin].copy()
            
            # Reset index to make the date a column for merging
            origin_weather.reset_index(inplace=True)
            if 'index' in origin_weather.columns:
                origin_weather.rename(columns={'index': 'date'}, inplace=True)
            
            # Convert date to date type (without time) for merging
            if 'date' in origin_weather.columns:
                origin_weather['date'] = pd.to_datetime(origin_weather['date']).dt.date
            
            # Merge flights with weather based on date
            merged_origin = pd.merge(
                origin_flights,
                origin_weather,
                left_on='DEP_DATE',
                right_on='date',
                how='left',
                suffixes=('', '_weather')
            )
            
            # Add information from origin to the merged DataFrame
            merged_dfs.append(merged_origin)
        else:
            print(f"Warning: No weather data for origin airport {origin}")
            # Still include flights without weather data
            merged_dfs.append(origin_flights)
    
    # Combine all merged DataFrames
    merged_data = pd.concat(merged_dfs, ignore_index=True)
    
    # Now merge with airport metadata
    merged_with_airport = pd.merge(
        merged_data,
        airport_data,
        left_on='ORIGIN',
        right_on='airport_code',
        how='left'
    )
    
    # Save the merged data
    merged_file_path = "merged_data/flight_weather_merged_2020.csv"
    merged_with_airport.to_csv(merged_file_path, index=False)
    
    print(f"Merged data saved to {merged_file_path}")
    print(f"Shape of merged data: {merged_with_airport.shape}")
    
    return merged_with_airport