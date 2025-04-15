from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from dataprofiler import Profiler

def profile_data_with_ydata(data, output_prefix="profiling_output/ydata"):
    """
    Profile the data using ydata-profiling (formerly pandas-profiling)
    
    Args:
        data (pandas.DataFrame): The dataset to profile
        output_prefix (str): Prefix for output files
        
    Returns:
        ProfileReport: The generated profile report
    """
    print("Profiling data with ydata-profiling...")
    
    # Configure the profiler with appropriate settings
    profile = ProfileReport(
        data, 
        title="Flight & Weather Data Profile (2020)",
        explorative=True,  # Enable more detailed profiling
        minimal=False,     # Don't use minimal mode
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        vars={
            "num": {
                "low_categorical_threshold": 0.05,
                "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]
            }
        }
    )
    
    # Save the profile report to HTML
    html_file = f"{output_prefix}_profile.html"
    profile.to_file(html_file)
    print(f"YData profile report saved to {html_file}")
    
    # Save the profile report to JSON
    json_file = f"{output_prefix}_profile.json"
    profile.to_json(json_file)
    print(f"YData profile data saved to {json_file}")
    
    return profile

def profile_data_with_dataprofiler(data, output_prefix="profiling_output/dataprofiler"):
    """
    Profile the data using DataProfiler
    
    Args:
        data (pandas.DataFrame): The dataset to profile
        output_prefix (str): Prefix for output files
        
    Returns:
        dict: The generated profile report
    """
    print("Profiling data with DataProfiler...")
    
    # Create a profiler instance
    profile = Profiler(data)
    
    # Generate a report with pretty formatting
    report = profile.report(report_options={"output_format": "pretty"})
    
    # Save the report to a JSON file
    json_file = f"{output_prefix}_profile.json"
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"DataProfiler report saved to {json_file}")
    
    return report

def profile_datasets(flight_data, weather_data_dict, airport_data, merged_data):
    """
    Profile all datasets using both profiling libraries
    
    Args:
        flight_data (pandas.DataFrame): Flight data
        weather_data_dict (dict): Dictionary of weather data by airport
        airport_data (pandas.DataFrame): Airport metadata
        merged_data (pandas.DataFrame): Merged dataset
        
    Returns:
        dict: Dictionary containing all profile reports
    """
    profiles = {}
    
    # Profile flight data
    print("\nProfiling flight data...")
    flight_ydata_profile = profile_data_with_ydata(flight_data, "profiling_output/flight_ydata")
    flight_dataprofiler_profile = profile_data_with_dataprofiler(flight_data, "profiling_output/flight_dataprofiler")
    profiles['flight'] = {
        'ydata': flight_ydata_profile,
        'dataprofiler': flight_dataprofiler_profile
    }
    
    # Profile a sample of weather data (first airport in the dictionary)
    if weather_data_dict:
        print("\nProfiling weather data (sample)...")
        first_airport = list(weather_data_dict.keys())[0]
        weather_sample = weather_data_dict[first_airport]
        weather_ydata_profile = profile_data_with_ydata(weather_sample, f"profiling_output/weather_{first_airport}_ydata")
        weather_dataprofiler_profile = profile_data_with_dataprofiler(weather_sample, f"profiling_output/weather_{first_airport}_dataprofiler")
        profiles['weather'] = {
            'ydata': weather_ydata_profile,
            'dataprofiler': weather_dataprofiler_profile
        }
    
    # Profile airport data
    print("\nProfiling airport metadata...")
    airport_ydata_profile = profile_data_with_ydata(airport_data, "profiling_output/airport_ydata")
    airport_dataprofiler_profile = profile_data_with_dataprofiler(airport_data, "profiling_output/airport_dataprofiler")
    profiles['airport'] = {
        'ydata': airport_ydata_profile,
        'dataprofiler': airport_dataprofiler_profile
    }
    
    # Profile merged data
    print("\nProfiling merged data...")
    merged_ydata_profile = profile_data_with_ydata(merged_data, "profiling_output/merged_ydata")
    merged_dataprofiler_profile = profile_data_with_dataprofiler(merged_data, "profiling_output/merged_dataprofiler")
    profiles['merged'] = {
        'ydata': merged_ydata_profile,
        'dataprofiler': merged_dataprofiler_profile
    }
    
    return profiles

def generate_summary_statistics(flight_data, weather_data_dict, airport_data, merged_data):
    """
    Generate summary statistics for all datasets
    
    Args:
        flight_data (pandas.DataFrame): Flight data
        weather_data_dict (dict): Dictionary of weather data by airport
        airport_data (pandas.DataFrame): Airport metadata
        merged_data (pandas.DataFrame): Merged dataset
        
    Returns:
        dict: Dictionary containing summary statistics
    """
    print("\nGenerating summary statistics...")
    
    summary = {}
    
    # Flight data statistics
    flight_stats = {
        "shape": flight_data.shape,
        "columns": list(flight_data.columns),
        "dtypes": {col: str(dtype) for col, dtype in flight_data.dtypes.items()},
        "missing_values": {col: int(flight_data[col].isna().sum()) for col in flight_data.columns},
        "missing_percentage": {col: float(flight_data[col].isna().mean() * 100) for col in flight_data.columns},
        "unique_values": {col: int(flight_data[col].nunique()) for col in flight_data.select_dtypes(include=['object']).columns},
        "numeric_stats": {col: {
            "min": float(flight_data[col].min()),
            "max": float(flight_data[col].max()),
            "mean": float(flight_data[col].mean()),
            "median": float(flight_data[col].median()),
            "std": float(flight_data[col].std())
        } for col in flight_data.select_dtypes(include=['float64', 'int64']).columns if not flight_data[col].isna().all()},
        "performance_stats": {
            "total_flights": len(flight_data),
            "on_time_flights": int(flight_data[(flight_data['ARR_DEL15'] == 0) & (flight_data['CANCELLED'] == 0) & (flight_data['DIVERTED'] == 0)].shape[0]),
            "delayed_flights": int(flight_data[flight_data['ARR_DEL15'] == 1].shape[0]),
            "cancelled_flights": int(flight_data[flight_data['CANCELLED'] == 1].shape[0]),
            "diverted_flights": int(flight_data[flight_data['DIVERTED'] == 1].shape[0]),
            "pct_on_time": float(flight_data[(flight_data['ARR_DEL15'] == 0) & (flight_data['CANCELLED'] == 0) & (flight_data['DIVERTED'] == 0)].shape[0] / len(flight_data) * 100),
            "pct_delayed": float(flight_data[flight_data['ARR_DEL15'] == 1].shape[0] / len(flight_data) * 100),
            "pct_cancelled": float(flight_data[flight_data['CANCELLED'] == 1].shape[0] / len(flight_data) * 100),
            "pct_diverted": float(flight_data[flight_data['DIVERTED'] == 1].shape[0] / len(flight_data) * 100),
            "avg_dep_delay": float(flight_data['DEP_DELAY'].mean()),
            "avg_arr_delay": float(flight_data['ARR_DELAY'].mean()),
        }
    }
    
    # Add delay reason statistics
    delay_cols = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    delayed_flights = flight_data[flight_data['ARR_DEL15'] == 1]
    
    if not delayed_flights.empty:
        total_delay_minutes = delayed_flights[delay_cols].sum().sum()
        flight_stats["delay_reasons"] = {
            "total_delay_minutes": float(total_delay_minutes)
        }
        
        for col in delay_cols:
            delay_minutes = delayed_flights[col].sum()
            flight_stats["delay_reasons"][col.lower()] = {
                "minutes": float(delay_minutes),
                "percentage": float(delay_minutes / total_delay_minutes * 100) if total_delay_minutes > 0 else 0
            }
    
    # Weather data statistics (sample from first airport)
    if weather_data_dict:
        first_airport = list(weather_data_dict.keys())[0]
        weather_sample = weather_data_dict[first_airport]
        
        weather_stats = {
            "shape": weather_sample.shape,
            "columns": list(weather_sample.columns),
            "dtypes": {col: str(dtype) for col, dtype in weather_sample.dtypes.items()},
            "missing_values": {col: int(weather_sample[col].isna().sum()) for col in weather_sample.columns},
            "missing_percentage": {col: float(weather_sample[col].isna().mean() * 100) for col in weather_sample.columns},
            "numeric_stats": {col: {
                "min": float(weather_sample[col].min()),
                "max": float(weather_sample[col].max()),
                "mean": float(weather_sample[col].mean()),
                "median": float(weather_sample[col].median()),
                "std": float(weather_sample[col].std())
            } for col in weather_sample.select_dtypes(include=['float64', 'int64']).columns if not weather_sample[col].isna().all()}
        }
    else:
        weather_stats = {"error": "No weather data available"}
    
    # Airport data statistics
    airport_stats = {
        "shape": airport_data.shape,
        "columns": list(airport_data.columns),
        "dtypes": {col: str(dtype) for col, dtype in airport_data.dtypes.items()},
        "missing_values": {col: int(airport_data[col].isna().sum()) for col in airport_data.columns},
        "missing_percentage": {col: float(airport_data[col].isna().mean() * 100) for col in airport_data.columns},
        "unique_values": {col: int(airport_data[col].nunique()) for col in airport_data.select_dtypes(include=['object']).columns},
        "numeric_stats": {col: {
            "min": float(airport_data[col].min()),
            "max": float(airport_data[col].max()),
            "mean": float(airport_data[col].mean()),
            "median": float(airport_data[col].median()),
            "std": float(airport_data[col].std())
        } for col in airport_data.select_dtypes(include=['float64', 'int64']).columns if not airport_data[col].isna().all()},
        "region_counts": airport_data['region'].value_counts().to_dict() if 'region' in airport_data.columns else {}
    }
    
    # Merged data statistics
    merged_stats = {
        "shape": merged_data.shape,
        "columns": list(merged_data.columns),
        "dtypes": {col: str(dtype) for col, dtype in merged_data.dtypes.items()},
        "missing_values": {col: int(merged_data[col].isna().sum()) for col in merged_data.columns},
        "missing_percentage": {col: float(merged_data[col].isna().mean() * 100) for col in merged_data.columns}
    }
    
    # Calculate correlations between weather and delays in merged data
    weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres', 'coco']
    delay_cols = ['DEP_DELAY', 'ARR_DELAY', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW']
    
    weather_delay_corr = {}
    
    for w_col in weather_cols:
        if w_col in merged_data.columns:
            for d_col in delay_cols:
                if d_col in merged_data.columns:
                    # Calculate correlation, ignoring NaN values
                    corr = merged_data[[w_col, d_col]].corr().iloc[0, 1]
                    weather_delay_corr[f"{w_col}_{d_col}"] = float(corr)
    
    merged_stats["weather_delay_correlations"] = weather_delay_corr
    
    # Combine all statistics
    summary = {
        "flight_data": flight_stats,
        "weather_data": weather_stats,
        "airport_data": airport_stats,
        "merged_data": merged_stats,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save summary to JSON
    summary_file = "profiling_output/summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary statistics saved to {summary_file}")
    
    # Print key statistics to console
    print("\nKey Flight Performance Statistics:")
    print(f"Total Flights: {flight_stats['performance_stats']['total_flights']}")
    print(f"On-time Flights: {flight_stats['performance_stats']['on_time_flights']} ({flight_stats['performance_stats']['pct_on_time']:.1f}%)")
    print(f"Delayed Flights: {flight_stats['performance_stats']['delayed_flights']} ({flight_stats['performance_stats']['pct_delayed']:.1f}%)")
    print(f"Cancelled Flights: {flight_stats['performance_stats']['cancelled_flights']} ({flight_stats['performance_stats']['pct_cancelled']:.1f}%)")
    print(f"Diverted Flights: {flight_stats['performance_stats']['diverted_flights']} ({flight_stats['performance_stats']['pct_diverted']:.1f}%)")
    
    if "delay_reasons" in flight_stats:
        print("\nDelay Reason Breakdown:")
        for reason, stats in flight_stats["delay_reasons"].items():
            if reason != "total_delay_minutes":
                reason_name = reason.replace("_", " ").title()
                print(f"{reason_name}: {stats['percentage']:.1f}%")
    
    if weather_delay_corr:
        print("\nTop Weather-Delay Correlations:")
        # Sort correlations by absolute value and print top 5
        sorted_corrs = sorted(weather_delay_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for pair, corr in sorted_corrs:
            w_col, d_col = pair.split('_', 1)
            print(f"{w_col} vs {d_col}: {corr:.3f}")
    
    return summary