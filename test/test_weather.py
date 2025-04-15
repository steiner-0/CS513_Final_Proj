#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import requests
import json
import io
from meteostat import Point, Daily, Hourly
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
#%%
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

location = Point(49.2497, -123.1193, 70)


data = Daily(location, start, end)
data = data.fetch()  # pd dataframe

print(data.head())

data.plot(y=['tavg', 'tmin', 'tmax'], figsize=(10, 6), title='Daily Temperature for 2018')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
#%%
start = datetime(2020, 1, 1)
end = datetime(2020, 1, 1, 23, 59)

# Get hourly data
data = Hourly('72219', start, end)
data = data.fetch()

# Print DataFrame
# print(data['coco'].isna().mean() * 100)
print(data['coco'])
# %%
TOKEN = "pTngtEgitntrkkENFsbABIHKoOdmStzK"

# Base URL for the NOAA CDO Web Services data endpoint
base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

# Using the hourly dataset identifier (commonly 'ISD' for Integrated Surface Data)
params = {
    "datasetid": "ISD",  # Use 'ISD' (or 'ISH' if your reference indicates)
    "stationid": "GHCND:USW00094846",  # Verify if this station id applies for hourly data
    "startdate": "2025-04-10",
    "enddate": "2025-04-10"
}

# Set up headers with your NOAA token
headers = {
    "token": TOKEN
}

# Make the GET request
response = requests.get(base_url, headers=headers, params=params)
response.raise_for_status()  # Raise an error for bad responses

# Parse the JSON response
data = response.json()

# Print the formatted JSON data for inspection
print(json.dumps(data, indent=2))