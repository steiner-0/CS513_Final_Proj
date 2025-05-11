#%%
import csv
import os
import re
import json
import sys
#%%
def extract_flight_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract origin airport and airline using regex
    origin_match = re.search(r'Origin Airport: .*?\((\w{3})\)', lines[1])
    airline_match = re.search(r'Airline: .*?\((\w{2})\)', lines[2])
    year_match = re.search(r'Year\(s\): (\d{4})', lines[5])

    origin_airport = origin_match.group(1) if origin_match else None
    airline = airline_match.group(1) if airline_match else None
    year = year_match.group(1) if year_match else None
    
    if not (origin_airport and airline and year):
        raise ValueError("Could not extract origin airport, airline, or year from the file.")

    # Process flight records starting from line 8
    reader = csv.DictReader(lines[7:], skipinitialspace=True)
    flights = []
    for row in reader:
        if not row["Date (MM/DD/YYYY)"]:
            break
        flights.append({
            "Date": row["Date (MM/DD/YYYY)"],
            "Tail Number": row["Tail Number"],
            "Destination Airport": row["Destination Airport"],
            "Delay Weather": row["Delay Weather (Minutes)"]
        })

    new_filename = f"{origin_airport.lower()}_{airline.lower()}_flight_delays_{year}.csv"
    new_filepath = os.path.join(os.path.dirname(file_path), new_filename)
    os.rename(file_path, new_filepath)

    return {
        "origin airport": origin_airport,
        "airline": airline,
        "year": year,
        "flights": flights
    }

# Example usage
result = extract_flight_info(r"C:\Users\yeyuc\Documents\CS513\CS513_Final_Proj\flight_data\Detailed_Statistics_Departures (14).csv")
json_preview = json.dumps(result, indent=2)
print(json_preview[:1000] ) # Truncate output for preview
# %%
print(result['flights'][-1])
# %%
