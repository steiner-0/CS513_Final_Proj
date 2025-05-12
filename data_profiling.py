import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import os
import io  # Add this import for StringIO

def load_data(filepath):
    """
    Load the flight and weather merged data.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def basic_exploration(df):
    """
    Perform basic data exploration.
    """
    print("\n=== BASIC DATA EXPLORATION ===")
    
    # Display basic info - FIX: Use StringIO instead of a list
    print("\nDataframe Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 
                                 'Percentage': missing_percent})
    missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False)
    
    print("\nMissing Values Analysis:")
    print(missing_data)
    
    return missing_data

def convert_datetime_fields(df):
    """
    Convert date and time fields to proper datetime format.
    """
    print("\n=== CONVERTING DATETIME FIELDS ===")
    
    # Convert FL_DATE to datetime if it's not already
    if 'FL_DATE' in df.columns and not pd.api.types.is_datetime64_dtype(df['FL_DATE']):
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
        print("Converted FL_DATE to datetime")
    
    # Convert DEP_DATETIME to datetime if it exists and is not already
    datetime_cols = ['CRS_DEP_DATETIME', 'CRS_ARR_DATETIME']
    for col in datetime_cols:
        if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
            print(f"Converted {col} to datetime")
        
    return df

def analyze_weather_delay(df):
    """
    Analyze the relationship between weather metrics and delay.
    """
    print("\n=== WEATHER DELAY ANALYSIS ===")
    
    # Check if WEATHER_DELAY exists in the dataframe
    if 'WEATHER_DELAY' not in df.columns:
        print("WEATHER_DELAY column not found in the dataset.")
        return df
    
    # Calculate average weather delay
    avg_weather_delay = df['WEATHER_DELAY'].mean()
    print(f"Average Weather Delay: {avg_weather_delay:.2f} minutes")
    
    # Count flights with weather delays
    flights_with_weather_delay = df[df['WEATHER_DELAY'] > 0].shape[0]
    total_flights = df.shape[0]
    percent_with_delay = (flights_with_weather_delay / total_flights) * 100
    print(f"Flights with Weather Delay: {flights_with_weather_delay} ({percent_with_delay:.2f}%)")
    
    # Create weather delay category
    df['delay_category'] = pd.cut(df['WEATHER_DELAY'], 
                                 bins=[-1, 0, 15, 30, 60, float('inf')],
                                 labels=['No Delay', 'Short (0-15min)', 'Medium (15-30min)', 
                                         'Long (30-60min)', 'Very Long (>60min)'])
    
    delay_distribution = df['delay_category'].value_counts().sort_index()
    print("\nDelay Category Distribution:")
    print(delay_distribution)
    
    return df

def create_visualizations(df, missing_data, output_dir):
    """
    Create visualizations for the dataset.
    """
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Figure 1: Missing values visualization
    plt.figure(figsize=(12, 6))
    if not missing_data.empty:
        missing_data = missing_data.sort_values('Percentage')
        plt.barh(missing_data.index, missing_data['Percentage'])
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Columns')
        plt.title('Missing Values Percentage by Column')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_values.png")
        print(f"Saved missing values visualization to {output_dir}/missing_values.png")
    else:
        print("No missing values to visualize.")
    
    # Figure 2: Weather delay distribution
    if 'WEATHER_DELAY' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['WEATHER_DELAY'].dropna(), bins=50, kde=True)
        plt.xlabel('Weather Delay (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Weather Delays')
        plt.xlim(0, df['WEATHER_DELAY'].quantile(0.95))  # Limit x-axis to 95th percentile
        plt.tight_layout()
        plt.savefig(f"{output_dir}/weather_delay_distribution.png")
        print(f"Saved weather delay distribution to {output_dir}/weather_delay_distribution.png")
    
    # Figure 3: Correlation between origin weather conditions and delay
    weather_cols = [col for col in df.columns if col.startswith('origin_') and df[col].dtype != 'object']
    if 'WEATHER_DELAY' in df.columns:
        weather_cols.append('WEATHER_DELAY')
    
    if len(weather_cols) > 1 and 'WEATHER_DELAY' in df.columns:
        plt.figure(figsize=(12, 10))
        # Use a sample of the data for faster correlation calculation if the dataset is large
        sample_size = min(100000, len(df))
        sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        try:
            correlation = sample_df[weather_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation between Origin Weather Conditions and Delay')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/weather_delay_correlation.png")
            print(f"Saved weather-delay correlation heatmap to {output_dir}/weather_delay_correlation.png")
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")
    
    # Figure 4: Boxplot of weather delay by origin
    if 'ORIGIN' in df.columns and 'WEATHER_DELAY' in df.columns:
        top_origins = df['ORIGIN'].value_counts().head(10).index
        origin_data = df[df['ORIGIN'].isin(top_origins)]
        
        # Use a sample for visualization if there's too much data
        if len(origin_data) > 50000:
            origin_data = origin_data.sample(50000, random_state=42)
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='ORIGIN', y='WEATHER_DELAY', data=origin_data)
        plt.xlabel('Origin Airport')
        plt.ylabel('Weather Delay (minutes)')
        plt.title('Weather Delay Distribution by Top 10 Origin Airports')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/delay_by_origin.png")
        print(f"Saved delay by origin boxplot to {output_dir}/delay_by_origin.png")
    
    plt.close('all')

def generate_profile_report(df, output_dir):
    """
    Generate a comprehensive profile report using ydata-profiling.
    """
    print("\n=== GENERATING PROFILE REPORT ===")
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # For large datasets, use minimal configuration and sample
        sample_size = min(100000, len(df))
        sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        # Generate the report with minimal configuration for large datasets
        profile = ProfileReport(sample_df, 
                               title="Flight Weather Merged Data Profiling Report",
                               minimal=True,  # Use minimal mode for large datasets
                               progress_bar=True)
        
        # Save the report
        profile.to_file(f"{output_dir}/profile_report.html")
        print(f"Saved comprehensive profile report to {output_dir}/profile_report.html")
        
        return profile
    except Exception as e:
        print(f"Error generating profile report: {e}")
        print("Continuing with other analyses...")
        return None

def analyze_weather_impact(df):
    """
    Analyze the impact of different weather conditions on delays.
    """
    print("\n=== ANALYZING WEATHER IMPACT ON DELAYS ===")
    
    if 'WEATHER_DELAY' not in df.columns:
        print("WEATHER_DELAY column not found in the dataset.")
        return None
    
    # Define weather condition columns to analyze
    weather_metrics = [
        ('Temperature', 'origin_temp'),
        ('Dew Point', 'origin_dwpt'),
        ('Relative Humidity', 'origin_rhum'),
        ('Precipitation', 'origin_prcp'),
        ('Wind Speed', 'origin_wspd'),
        ('Pressure', 'origin_pres'),
        ('Weather Condition Code', 'origin_coco')
    ]
    
    # Create a table of correlations
    correlations = []
    
    # For large datasets, use a sample for faster correlation calculation
    sample_size = min(100000, len(df))
    sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    for name, col in weather_metrics:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr = sample_df[['WEATHER_DELAY', col]].corr().iloc[0, 1]
                correlations.append((name, col, corr))
            except Exception as e:
                print(f"Could not calculate correlation for {name}: {e}")
    
    if correlations:
        correlations_df = pd.DataFrame(correlations, columns=['Weather Metric', 'Column', 'Correlation with Weather Delay'])
        correlations_df = correlations_df.sort_values('Correlation with Weather Delay', ascending=False)
        
        print("\nCorrelation between Weather Metrics and Delay:")
        print(correlations_df)
        
        # Identify strongest relationships
        strongest_positive = correlations_df.loc[correlations_df['Correlation with Weather Delay'] == 
                                               correlations_df['Correlation with Weather Delay'].max()]
        strongest_negative = correlations_df.loc[correlations_df['Correlation with Weather Delay'] == 
                                               correlations_df['Correlation with Weather Delay'].min()]
        
        print(f"\nStrongest positive correlation: {strongest_positive['Weather Metric'].values[0]} "
              f"({strongest_positive['Correlation with Weather Delay'].values[0]:.3f})")
        print(f"Strongest negative correlation: {strongest_negative['Weather Metric'].values[0]} "
              f"({strongest_negative['Correlation with Weather Delay'].values[0]:.3f})")
        
        return correlations_df
    else:
        print("No numeric weather metrics found for correlation analysis.")
        return None

def generate_summary_report(df, output_dir, weather_impact=None):
    """
    Generate a summary report with key findings.
    """
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open file for writing
    with open(f"{output_dir}/summary_report.txt", "w") as file:
        # Write header
        file.write("=== FLIGHT-WEATHER DATA PROFILING SUMMARY REPORT ===\n\n")
        file.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Dataset: Flight-Weather Merged Data\n")
        file.write(f"Records: {df.shape[0]}\n")
        file.write(f"Variables: {df.shape[1]}\n\n")
        
        # Dataset summary
        file.write("== DATASET SUMMARY ==\n")
        if 'FL_DATE' in df.columns:
            try:
                if not pd.api.types.is_datetime64_dtype(df['FL_DATE']):
                    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
                file.write(f"Date range: {df['FL_DATE'].min()} to {df['FL_DATE'].max()}\n")
            except Exception as e:
                file.write(f"Could not determine date range: {e}\n")
        
        if 'ORIGIN' in df.columns:
            top_origins = df['ORIGIN'].value_counts().head(5)
            file.write("\nTop 5 Origin Airports:\n")
            for airport, count in top_origins.items():
                file.write(f"  - {airport}: {count} flights ({count/df.shape[0]*100:.1f}%)\n")
        
        # Missing data summary
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        file.write("\n== MISSING DATA SUMMARY ==\n")
        if len(missing_cols) > 0:
            file.write(f"Columns with missing values: {len(missing_cols)}\n")
            for col, count in missing_cols.items():
                file.write(f"  - {col}: {count} missing values ({count/df.shape[0]*100:.1f}%)\n")
        else:
            file.write("No missing values found in the dataset.\n")
        
        # Weather delay summary
        if 'WEATHER_DELAY' in df.columns:
            file.write("\n== WEATHER DELAY SUMMARY ==\n")
            avg_delay = df['WEATHER_DELAY'].mean()
            max_delay = df['WEATHER_DELAY'].max()
            delays_count = df[df['WEATHER_DELAY'] > 0].shape[0]
            
            file.write(f"Average weather delay: {avg_delay:.2f} minutes\n")
            file.write(f"Maximum weather delay: {max_delay:.2f} minutes\n")
            file.write(f"Flights with weather delays: {delays_count} ({delays_count/df.shape[0]*100:.1f}%)\n")
        
        # Weather correlations summary
        if weather_impact is not None and not weather_impact.empty:
            file.write("\n== WEATHER-DELAY CORRELATION SUMMARY ==\n")
            for _, row in weather_impact.head(5).iterrows():
                metric = row['Weather Metric']
                corr = row['Correlation with Weather Delay']
                file.write(f"  - {metric}: {corr:.3f}\n")
        
        # Conclusion
        file.write("\n== CONCLUSION ==\n")
        file.write("This report provides a summary of the flight-weather merged dataset. ")
        file.write("For more detailed analysis, refer to the visualizations in the profiling_output directory.\n")
    
    print(f"Saved summary report to {output_dir}/summary_report.txt")

def main():
    # Set input and output paths
    input_filepath = "output/flight_weather_merged.csv"
    output_dir = "profiling_output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load the data
        df = load_data(input_filepath)
        
        # Basic exploration
        missing_data = basic_exploration(df)
        
        # Convert datetime fields
        df = convert_datetime_fields(df)
        
        # Analyze weather delay
        df = analyze_weather_delay(df)
        
        # Analyze weather impact
        weather_impact = analyze_weather_impact(df)
        
        # Generate visualizations
        create_visualizations(df, missing_data, output_dir)
        
        # Generate profile report - this can be resource-intensive
        try:
            profile = generate_profile_report(df, output_dir)
        except Exception as e:
            print(f"Error generating profile report: {e}")
            print("Continuing with other analyses...")
        
        # Generate summary report
        generate_summary_report(df, output_dir, weather_impact)
        
        print(f"\nData profiling complete. All outputs saved to {output_dir}/")
    
    except Exception as e:
        print(f"Error during data profiling: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for better debugging

if __name__ == "__main__":
    main()