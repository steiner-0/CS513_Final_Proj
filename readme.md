# Flight Weather Delay Analysis

This project analyzes the relationship between weather conditions and flight delays using data from weather stations and flight performance records.

## Project Overview

The project consists of three main components:

1. **Data Pipeline**: Retrieves flight data and weather data from various sources, processes them, and merges them into a single dataset.
2. **Data Profiling**: Analyzes the merged dataset to extract meaningful insights about weather conditions and delays.
3. **Predictive Modeling**: Implements a logistic regression model to predict flight delays based on weather conditions.

## Requirements

- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - meteostat
  - tqdm
  - ydata-profiling
  - plotly (for some visualizations)

You can install all required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data_pipeline.py        # Main data retrieval and processing script
├── data_profiling.py       # Complete dataset profiling script
├── DataProfiling.ipynb     # Jupyter notebook for sample data exploration
├── logistic_regression.py  # Predictive modeling script
├── coco_analysis.py        # Analysis of weather condition codes
├── coco.txt                # Weather condition code mappings
├── flight_data/            # Directory to store flight data files
├── airport_data/           # Directory to store airport data
│   └── airports.dat        # Airport database file
├── weather_data/           # Directory for cached weather data
├── output/                 # Directory for merged data output
├── profiling_output/       # Directory for profiling results
└── coco_analysis_output/   # Directory for condition code analysis results
```

## Running the Project

### Step 1: Setup Directories

Create the necessary directories for the project:

```bash
mkdir -p flight_data airport_data weather_data output profiling_output coco_analysis_output
```

### Step 2: Download Required Data

1. Download the airport database file and place it in the `airport_data` directory
2. Place your flight data CSV files in the `flight_data` directory

### Step 3: Run the Data Pipeline

The data pipeline retrieves, processes, and merges flight and weather data:

```bash
python data_pipeline.py
```

This script will:
- Load flight data from CSV files in the `flight_data` directory
- Fetch weather data for each airport using the Meteostat API
- Merge flight and weather data
- Save the merged data to `output/flight_weather_merged.csv`

### Step 4: Run Data Profiling

After generating the merged dataset, run the data profiling script:

```bash
python data_profiling.py
```

This will:
- Generate statistical summaries of the dataset
- Create visualizations of key relationships
- Save a comprehensive profile report to `profiling_output/`

For a deeper analysis of weather condition codes:

```bash
python coco_analysis.py
```

This script analyzes the relationship between specific weather conditions (using their codes) and flight delays, generating results in the `coco_analysis_output/` directory.

### Step 5: Run Predictive Modeling

Finally, run the logistic regression model to predict flight delays:

```bash
python logistic_regression.py
```

This script:
- Builds a logistic regression model to predict whether a flight will experience a weather delay
- Evaluates model performance
- Analyzes which weather features are most important for predicting delays
- Outputs feature importance plots and classification metrics

## Notes on Data Handling

- Missing weather data is handled according to rules in `null_value_handle.txt`
- Weather condition codes (coco) are mapped to human-readable conditions in `coco.txt`
- The data pipeline makes several replacements to handle missing values, including:
  - Missing weather condition codes (coco) are filled with 1 (Clear)
  - Missing precipitation and snow values are filled with 0
  - Wind gust speed (wpgt) values of 0 are replaced with wind speed (wspd) when available

## Results

After running the full pipeline, you'll find:
- A merged dataset in the `output` directory
- Profiling visualizations and reports in the `profiling_output` directory
- Weather condition analysis in the `coco_analysis_output` directory
- Feature importance plots and model evaluation metrics in the project root

The analysis shows which weather conditions most strongly correlate with flight delays and evaluates the predictive power of various weather metrics.
