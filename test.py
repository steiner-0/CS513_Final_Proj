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
start = datetime(2013, 1, 1)
end = datetime(2013, 12, 31, 23, 59)

# Get hourly data
data = Hourly('72219', start, end)
data = data.fetch()

# Print DataFrame
print(data['coco'].isna().mean() * 100)
