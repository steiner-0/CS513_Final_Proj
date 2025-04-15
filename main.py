from airport_data import *
from flight_data import *
from weather_data import *
from data_profiling import *
from utils import *

def main():
    """
    Main execution function
    """
    print("Starting data processing and profiling for 2020 flight and weather data...")
    
    # 1. ACQUIRE DATA
    # Generate list of airports (this would normally come from a reference dataset)
    airports = ['ATL', 'ORD', 'LAX', 'DFW', 'JFK', 'DEN', 'SFO', 'SEA', 'LAS', 'MCO', 
                'BOS', 'PHX', 'MIA', 'IAH', 'MSP', 'DTW', 'PHL', 'LGA', 'CLT', 'BWI']
    
    # 1.1 Download flight data for 2020
    flight_data_raw = download_flight_data_for_year(2020)
    
    # 1.2 Download weather data for 2020
    weather_data_raw = download_weather_data(2020, airports)
    
    # 1.3 Create airport metadata
    airport_data_raw = create_airport_metadata(airports)
    
    # 2. CLEAN DATA
    # 2.1 Clean flight data
    flight_data_clean = clean_flight_data(flight_data_raw)
    
    # 2.2 Clean weather data
    weather_data_clean = clean_weather_data(weather_data_raw)
    
    # 2.3 Clean airport metadata
    airport_data_clean = clean_airport_metadata(airport_data_raw)
    
    # 3. MERGE DATA
    # Merge flight and weather data
    merged_data = merge_flight_and_weather(flight_data_clean, weather_data_clean, airport_data_clean)
    
    # 4. DATA PROFILING AND STATISTICS
    # 4.1 Profile all datasets
    profiles = profile_datasets(flight_data_clean, weather_data_clean, airport_data_clean, merged_data)
    
    # 4.2 Generate summary statistics
    summary = generate_summary_statistics(flight_data_clean, weather_data_clean, airport_data_clean, merged_data)
    
    print("\nData processing and profiling complete!")
    print("Summary reports and statistics are available in the profiling_output directory.")

if __name__ == "__main__":
    main()