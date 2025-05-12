"""
coco_analysis.py - Analysis of Weather Condition Codes and Flight Delays

This script analyzes the relationship between weather condition codes (coco) at
both origin and destination airports and their impact on flight weather delays.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# Set up styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(filepath='output/flight_weather_merged.csv'):
    """
    Load the flight-weather merged data.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def load_coco_mapping(filepath='coco.txt'):
    """
    Load the weather condition code mapping from the text file.
    """
    print(f"Loading coco mapping from {filepath}...")
    coco_map = {}
    
    try:
        with open(filepath, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    code = int(parts[0])
                    condition = parts[1]
                    coco_map[code] = condition
    except Exception as e:
        print(f"Error loading coco mapping: {e}")
        # Create a fallback mapping based on the data in the prompt
        coco_map = {
            1: "Clear", 2: "Fair", 3: "Cloudy", 4: "Overcast", 5: "Fog",
            6: "Freezing Fog", 7: "Light Rain", 8: "Rain", 9: "Heavy Rain",
            10: "Freezing Rain", 11: "Heavy Freezing Rain", 12: "Sleet",
            13: "Heavy Sleet", 14: "Light Snowfall", 15: "Snowfall",
            16: "Heavy Snowfall", 17: "Rain Shower", 18: "Heavy Rain Shower",
            19: "Sleet Shower", 20: "Heavy Sleet Shower", 21: "Snow Shower",
            22: "Heavy Snow Shower", 23: "Lightning", 24: "Hail",
            25: "Thunderstorm", 26: "Heavy Thunderstorm", 27: "Storm"
        }
    
    print(f"Loaded {len(coco_map)} weather condition codes")
    return coco_map

def prepare_data(df, coco_map):
    """
    Prepare the data for analysis by:
    1. Converting coco codes to integers
    2. Adding condition labels
    3. Creating delay categories
    """
    print("Preparing data for analysis...")
    
    # Make a copy to avoid modifying the original dataframe
    df_analysis = df.copy()
    
    # Ensure coco columns are integers (they might be floats in the dataset)
    for col in ['origin_coco', 'dest_coco']:
        if col in df_analysis.columns:
            # Handle any NaN values
            df_analysis[col] = df_analysis[col].fillna(1)  # Fill with "Clear" as per the data pipeline
            df_analysis[col] = df_analysis[col].astype(int)
            
            # Add weather condition labels
            label_col = col.replace('coco', 'condition')
            df_analysis[label_col] = df_analysis[col].map(coco_map)
            
            # For any codes not in our mapping, use "Unknown"
            df_analysis[label_col] = df_analysis[label_col].fillna("Unknown")
    
    # Create delay categories for easier analysis
    if 'WEATHER_DELAY' in df_analysis.columns:
        df_analysis['delay_category'] = pd.cut(
            df_analysis['WEATHER_DELAY'], 
            bins=[-1, 0, 15, 45, 120, float('inf')],
            labels=['No Delay', 'Short (0-15min)', 'Medium (15-45min)', 
                    'Long (45-120min)', 'Very Long (>120min)']
        )
    
    # Flag flights with any weather delay
    df_analysis['has_delay'] = df_analysis['WEATHER_DELAY'] > 0
    
    print("Data preparation complete.")
    return df_analysis

def analyze_coco_delay_relationship(df, coco_map, output_dir='coco_analysis_output'):
    """
    Analyze the relationship between weather condition codes and delays.
    """
    print("\n=== ANALYZING COCO-DELAY RELATIONSHIP ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Prepare the results dataframe
    results = []
    
    # Analyze origin and destination cocos
    for location in ['origin', 'dest']:
        coco_col = f'{location}_coco'
        condition_col = f'{location}_condition'
        
        if coco_col not in df.columns or 'WEATHER_DELAY' not in df.columns:
            print(f"Required columns {coco_col} or WEATHER_DELAY not found in the dataset.")
            continue
            
        print(f"\nAnalyzing {location.capitalize()} Weather Conditions:")
        
        # Get the distribution of weather conditions
        condition_counts = df[condition_col].value_counts()
        total_flights = len(df)
        
        print(f"\n{location.capitalize()} Weather Condition Distribution:")
        for condition, count in condition_counts.items():
            percentage = (count / total_flights) * 100
            print(f"  - {condition}: {count} flights ({percentage:.2f}%)")
        
        # Calculate average delay by weather condition
        avg_delay_by_condition = df.groupby(condition_col)['WEATHER_DELAY'].mean().sort_values(ascending=False)
        
        print(f"\nAverage Weather Delay by {location.capitalize()} Weather Condition:")
        for condition, avg_delay in avg_delay_by_condition.items():
            print(f"  - {condition}: {avg_delay:.2f} minutes")
        
        # Calculate percentage of flights with delays by weather condition
        delay_pct_by_condition = df.groupby(condition_col)['has_delay'].mean() * 100
        delay_pct_by_condition = delay_pct_by_condition.sort_values(ascending=False)
        
        print(f"\nPercentage of Flights with Weather Delays by {location.capitalize()} Weather Condition:")
        for condition, pct in delay_pct_by_condition.items():
            print(f"  - {condition}: {pct:.2f}%")
        
        # Collect results for each condition
        for coco, condition in coco_map.items():
            condition_data = df[df[coco_col] == coco]
            if len(condition_data) == 0:
                continue
                
            avg_delay = condition_data['WEATHER_DELAY'].mean()
            max_delay = condition_data['WEATHER_DELAY'].max()
            delay_pct = (condition_data['has_delay'].sum() / len(condition_data)) * 100
            
            results.append({
                'Location': location.capitalize(),
                'COCO': coco,
                'Weather Condition': condition,
                'Flight Count': len(condition_data),
                'Average Delay (min)': avg_delay,
                'Maximum Delay (min)': max_delay,
                'Flights with Delay (%)': delay_pct
            })
    
    # Create a results dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/coco_delay_analysis.csv", index=False)
    print(f"\nResults saved to {output_dir}/coco_delay_analysis.csv")
    
    return results_df

def create_visualizations(df, results_df, coco_map, output_dir='coco_analysis_output'):
    """
    Create visualizations for the coco-delay relationship.
    """
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Barplot of average delay by origin weather condition
    plt.figure(figsize=(14, 10))
    origin_results = results_df[results_df['Location'] == 'Origin'].sort_values('Average Delay (min)', ascending=False)
    origin_results = origin_results[origin_results['Flight Count'] >= 1000]  # Filter for statistical significance
    
    sns.barplot(x='Weather Condition', y='Average Delay (min)', data=origin_results)
    plt.title('Average Weather Delay by Origin Weather Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/origin_coco_avg_delay.png")
    print(f"Saved origin coco average delay plot to {output_dir}/origin_coco_avg_delay.png")
    
    # 2. Barplot of average delay by destination weather condition
    plt.figure(figsize=(14, 10))
    dest_results = results_df[results_df['Location'] == 'Dest'].sort_values('Average Delay (min)', ascending=False)
    dest_results = dest_results[dest_results['Flight Count'] >= 1000]  # Filter for statistical significance
    
    sns.barplot(x='Weather Condition', y='Average Delay (min)', data=dest_results)
    plt.title('Average Weather Delay by Destination Weather Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dest_coco_avg_delay.png")
    print(f"Saved destination coco average delay plot to {output_dir}/dest_coco_avg_delay.png")
    
    # 3. Heatmap showing delay percentage for different origin-destination weather condition combinations
    pivot_data = create_origin_dest_pivot(df)
    if pivot_data is not None:
        plt.figure(figsize=(16, 14))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
        plt.title('Percentage of Flights with Weather Delays by Origin-Destination Weather Condition')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/origin_dest_coco_heatmap.png")
        print(f"Saved origin-destination coco heatmap to {output_dir}/origin_dest_coco_heatmap.png")
    
    # 4. Barplot of percentage of flights with delays by weather condition
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Weather Condition', y='Flights with Delay (%)', 
                data=origin_results.sort_values('Flights with Delay (%)', ascending=False))
    plt.title('Percentage of Flights with Weather Delays by Origin Weather Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/origin_coco_delay_percentage.png")
    print(f"Saved origin coco delay percentage plot to {output_dir}/origin_coco_delay_percentage.png")
    
    # 5. Comparative plot of origin vs destination weather impact
    plot_origin_vs_dest_impact(results_df, output_dir)
    
    # 6. Distribution of delay lengths by most impactful weather conditions
    plot_delay_distribution_by_condition(df, output_dir)
    
    plt.close('all')
    print("Visualization creation complete.")

def create_origin_dest_pivot(df):
    """
    Create a pivot table showing the percentage of flights with delays for 
    different origin-destination weather condition combinations.
    """
    if 'origin_condition' not in df.columns or 'dest_condition' not in df.columns:
        return None
        
    # Get the most common weather conditions for better visualization
    origin_conditions = df['origin_condition'].value_counts().head(8).index
    dest_conditions = df['dest_condition'].value_counts().head(8).index
    
    # Filter data to include only common conditions
    filtered_df = df[df['origin_condition'].isin(origin_conditions) & 
                      df['dest_condition'].isin(dest_conditions)]
    
    # Create pivot table
    pivot = pd.pivot_table(
        filtered_df,
        values='has_delay',
        index='origin_condition',
        columns='dest_condition',
        aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
    )
    
    return pivot

def plot_origin_vs_dest_impact(results_df, output_dir):
    """
    Create a comparative plot showing the impact of origin vs destination
    weather conditions on delays.
    """
    # Merge origin and destination data for comparison
    origin_data = results_df[results_df['Location'] == 'Origin'].copy()
    dest_data = results_df[results_df['Location'] == 'Dest'].copy()
    
    if origin_data.empty or dest_data.empty:
        return
    
    # Get common weather conditions
    common_conditions = list(set(origin_data['Weather Condition']) & 
                              set(dest_data['Weather Condition']))
    
    # Filter for common conditions with sufficient data
    origin_filtered = origin_data[origin_data['Weather Condition'].isin(common_conditions) & 
                                   (origin_data['Flight Count'] >= 1000)]
    dest_filtered = dest_data[dest_data['Weather Condition'].isin(common_conditions) & 
                                (dest_data['Flight Count'] >= 1000)]
    
    # Create a new dataframe for plotting
    plot_data = []
    for condition in common_conditions:
        origin_row = origin_filtered[origin_filtered['Weather Condition'] == condition]
        dest_row = dest_filtered[dest_filtered['Weather Condition'] == condition]
        
        if not origin_row.empty and not dest_row.empty:
            plot_data.append({
                'Weather Condition': condition,
                'Origin Avg Delay': origin_row['Average Delay (min)'].values[0],
                'Dest Avg Delay': dest_row['Average Delay (min)'].values[0],
                'Origin Delay %': origin_row['Flights with Delay (%)'].values[0],
                'Dest Delay %': dest_row['Flights with Delay (%)'].values[0]
            })
    
    if not plot_data:
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by maximum impact (either origin or destination)
    plot_df['Max Delay'] = plot_df[['Origin Avg Delay', 'Dest Avg Delay']].max(axis=1)
    plot_df = plot_df.sort_values('Max Delay', ascending=False)
    
    # Create a comparative bar plot
    plt.figure(figsize=(14, 10))
    bar_width = 0.35
    index = np.arange(len(plot_df))
    
    plt.bar(index, plot_df['Origin Avg Delay'], bar_width, label='Origin Airport')
    plt.bar(index + bar_width, plot_df['Dest Avg Delay'], bar_width, label='Destination Airport')
    
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Weather Delay (minutes)')
    plt.title('Impact of Weather Conditions: Origin vs Destination Airport')
    plt.xticks(index + bar_width / 2, plot_df['Weather Condition'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/origin_vs_dest_impact.png")
    print(f"Saved origin vs destination impact plot to {output_dir}/origin_vs_dest_impact.png")
    
    # Create a delay percentage comparison plot
    plt.figure(figsize=(14, 10))
    plt.bar(index, plot_df['Origin Delay %'], bar_width, label='Origin Airport')
    plt.bar(index + bar_width, plot_df['Dest Delay %'], bar_width, label='Destination Airport')
    
    plt.xlabel('Weather Condition')
    plt.ylabel('Percentage of Flights with Weather Delays')
    plt.title('Flights with Weather Delays: Origin vs Destination Weather Conditions')
    plt.xticks(index + bar_width / 2, plot_df['Weather Condition'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/origin_vs_dest_delay_percentage.png")
    print(f"Saved origin vs destination delay percentage plot to {output_dir}/origin_vs_dest_delay_percentage.png")

def plot_delay_distribution_by_condition(df, output_dir):
    """
    Plot the distribution of delay lengths for the most impactful weather conditions.
    """
    if 'origin_condition' not in df.columns or 'WEATHER_DELAY' not in df.columns:
        return
    
    # Find top 5 conditions with highest average delays
    top_conditions = df.groupby('origin_condition')['WEATHER_DELAY'].mean().nlargest(5).index
    
    # Filter for flights with actual delays
    delay_df = df[df['WEATHER_DELAY'] > 0]
    
    # Filter for top conditions
    filtered_df = delay_df[delay_df['origin_condition'].isin(top_conditions)]
    
    if filtered_df.empty:
        return
    
    # Create violin plots
    plt.figure(figsize=(16, 10))
    sns.violinplot(x='origin_condition', y='WEATHER_DELAY', data=filtered_df, cut=0)
    plt.title('Distribution of Weather Delay Duration by Origin Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Weather Delay (minutes)')
    plt.ylim(0, filtered_df['WEATHER_DELAY'].quantile(0.95))  # Limit y-axis for better visualization
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_distribution_by_condition.png")
    print(f"Saved delay distribution plot to {output_dir}/delay_distribution_by_condition.png")
    
    # Create boxplots for comparison
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='origin_condition', y='WEATHER_DELAY', data=filtered_df)
    plt.title('Weather Delay Duration by Origin Weather Condition (Boxplot)')
    plt.xlabel('Weather Condition')
    plt.ylabel('Weather Delay (minutes)')
    plt.ylim(0, filtered_df['WEATHER_DELAY'].quantile(0.95))  # Limit y-axis for better visualization
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_boxplot_by_condition.png")
    print(f"Saved delay boxplot to {output_dir}/delay_boxplot_by_condition.png")

def generate_summary_report(results_df, output_dir='coco_analysis_output'):
    """
    Generate a summary report with key findings.
    """
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open file for writing
    with open(f"{output_dir}/coco_analysis_summary.txt", "w") as file:
        # Write header
        file.write("=== WEATHER CONDITION CODE (COCO) ANALYSIS SUMMARY REPORT ===\n\n")
        file.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Most impactful origin weather conditions
        file.write("== TOP 5 ORIGIN WEATHER CONDITIONS WITH HIGHEST AVERAGE DELAYS ==\n")
        origin_top = results_df[results_df['Location'] == 'Origin'].sort_values('Average Delay (min)', ascending=False).head(5)
        
        for _, row in origin_top.iterrows():
            file.write(f"  - {row['Weather Condition']} (Code {row['COCO']}): ")
            file.write(f"{row['Average Delay (min)']:.2f} minutes average delay, ")
            file.write(f"{row['Flights with Delay (%)']:.2f}% of {row['Flight Count']} flights delayed\n")
        
        # Most impactful destination weather conditions
        file.write("\n== TOP 5 DESTINATION WEATHER CONDITIONS WITH HIGHEST AVERAGE DELAYS ==\n")
        dest_top = results_df[results_df['Location'] == 'Dest'].sort_values('Average Delay (min)', ascending=False).head(5)
        
        for _, row in dest_top.iterrows():
            file.write(f"  - {row['Weather Condition']} (Code {row['COCO']}): ")
            file.write(f"{row['Average Delay (min)']:.2f} minutes average delay, ")
            file.write(f"{row['Flights with Delay (%)']:.2f}% of {row['Flight Count']} flights delayed\n")
        
        # Compare origin vs destination impact
        file.write("\n== ORIGIN VS DESTINATION WEATHER IMPACT ==\n")
        
        # Calculate average impact by location
        avg_by_location = results_df.groupby('Location')['Average Delay (min)'].mean()
        pct_by_location = results_df.groupby('Location')['Flights with Delay (%)'].mean()
        
        file.write(f"Origin airport weather conditions: {avg_by_location.get('Origin', 0):.2f} minutes average delay, ")
        file.write(f"{pct_by_location.get('Origin', 0):.2f}% flights with delays\n")
        
        file.write(f"Destination airport weather conditions: {avg_by_location.get('Dest', 0):.2f} minutes average delay, ")
        file.write(f"{pct_by_location.get('Dest', 0):.2f}% flights with delays\n")
        
        # Calculate the most delay-inducing conditions
        file.write("\n== WEATHER CONDITIONS RANKED BY DELAY IMPACT ==\n")
        
        # Get conditions present in both origin and destination
        common_conditions = set(
            results_df[results_df['Location'] == 'Origin']['Weather Condition'].unique()
        ).intersection(
            results_df[results_df['Location'] == 'Dest']['Weather Condition'].unique()
        )
        
        # Calculate average impact across origin and destination for each condition
        condition_impact = defaultdict(lambda: {'delay': 0, 'percentage': 0, 'count': 0})
        
        for _, row in results_df.iterrows():
            condition = row['Weather Condition']
            condition_impact[condition]['delay'] += row['Average Delay (min)']
            condition_impact[condition]['percentage'] += row['Flights with Delay (%)']
            condition_impact[condition]['count'] += 1
        
        # Convert to list and sort by average delay
        impact_list = [{
            'condition': condition,
            'avg_delay': data['delay'] / data['count'],
            'avg_percentage': data['percentage'] / data['count']
        } for condition, data in condition_impact.items()]
        
        impact_list.sort(key=lambda x: x['avg_delay'], reverse=True)
        
        for item in impact_list[:10]:  # Top 10 conditions
            file.write(f"  - {item['condition']}: {item['avg_delay']:.2f} minutes average delay, ")
            file.write(f"{item['avg_percentage']:.2f}% of flights delayed\n")
        
        # Conclusions
        file.write("\n== CONCLUSIONS ==\n")
        file.write("1. The weather conditions with the most significant impact on flight delays are:\n")
        for item in impact_list[:3]:  # Top 3 conditions
            file.write(f"   - {item['condition']}\n")
        
        # Compare origin vs dest
        if avg_by_location.get('Origin', 0) > avg_by_location.get('Dest', 0):
            file.write("\n2. Weather conditions at the origin airport tend to cause longer delays than at the destination.\n")
        else:
            file.write("\n2. Weather conditions at the destination airport tend to cause longer delays than at the origin.\n")
        
        file.write("\n3. The majority of weather delays occur when these specific conditions are present:\n")
        pct_sorted = sorted(impact_list, key=lambda x: x['avg_percentage'], reverse=True)
        for item in pct_sorted[:3]:  # Top 3 by percentage
            file.write(f"   - {item['condition']} ({item['avg_percentage']:.2f}% of flights delayed)\n")
        
        file.write("\nThis analysis provides insights into how different weather conditions at origin and ")
        file.write("destination airports affect flight delays. For more detailed visualizations, refer to ")
        file.write("the plots in the coco_analysis_output directory.\n")
    
    print(f"Saved summary report to {output_dir}/coco_analysis_summary.txt")

def main():
    """Main function to run the analysis"""
    # Set output directory
    output_dir = "coco_analysis_output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Step 1: Load the flight-weather merged data
        df = load_data()
        
        # Step 2: Load the coco mapping
        coco_map = load_coco_mapping()
        
        # Step 3: Prepare the data for analysis
        df_analysis = prepare_data(df, coco_map)
        
        # Step 4: Analyze the relationship between cocos and delays
        results_df = analyze_coco_delay_relationship(df_analysis, coco_map, output_dir)
        
        # Step 5: Create visualizations
        create_visualizations(df_analysis, results_df, coco_map, output_dir)
        
        # Step 6: Generate a summary report
        generate_summary_report(results_df, output_dir)
        
        print(f"\nCoco analysis complete. All outputs saved to {output_dir}/")
    
    except Exception as e:
        print(f"Error during coco analysis: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for better debugging

if __name__ == "__main__":
    main()