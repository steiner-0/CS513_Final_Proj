#%%
import csv
from meteostat import Stations
#%%
def get_airport_coordinates(iata_code, filepath='airports.dat'):
    """
    Looks up the latitude and longitude of an airport given its IATA code.
    
    Parameters:
        iata_code (str): The 3-letter IATA airport code.
        filepath (str): Path to the OpenFlights airports.dat file.
        
    Returns:
        tuple: (latitude, longitude) if found, otherwise None.
    """
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Columns: ID, Name, City, Country, IATA, ICAO, Latitude, Longitude, ...
            if len(row) > 7 and row[4].upper() == iata_code.upper():
                return float(row[6]), float(row[7])
            
    raise ValueError(f"Airport with IATA code {iata_code} not found in {filepath}.")

# %%
get_airport_coordinates("SFO", filepath=r'C:\Users\yeyuc\Documents\CS513\CS513_Final_Proj\airport_data\airports.dat')
# %%

def airportloc2station(iata_code, filepath='airports.dat'):
    """
    Converts an airport IATA code to a Meteostat station ID.
    
    Parameters:
        iata_code (str): The 3-letter IATA airport code.
        filepath (str): Path to the OpenFlights airports.dat file.
        
    Returns:
        str: Meteostat station ID if found, otherwise None.
    """
    try:
        latitude, longitude = get_airport_coordinates(iata_code, filepath)
    except ValueError as e:
        print(e)
        return None

    # Use Meteostat to find nearby stations
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    
    # Fetch the nearest station
    nearest_station = stations.fetch(1)
    
    return nearest_station.index[0] if not nearest_station.empty else None
airportloc2station("SFO", filepath=r'C:\Users\yeyuc\Documents\CS513\CS513_Final_Proj\airport_data\airports.dat')
# %%
