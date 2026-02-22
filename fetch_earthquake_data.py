import requests
import json
import pandas as pd

url='https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2024-01-01&endtime=2024-12-31'

params= {
        'format': 'geojson',
        'starttime': '2024-01-01',
        'endtime': '2024-12-31',
        'minmagnitude': 4
    }

response = requests.get(url, params=params)
data = response.json()

#print the keys and the data
print(data.keys())
##print(json.dumps(data, indent=2))
print(len(data['features']))

# extract features and convert to DataFrame
features = data['features']
df = pd.json_normalize(features)
df.to_csv('earthquake_data.csv', index=False, encoding='utf-8-sig')

print(f"\n there is {len(df)} data in total")
print(f"\nDataFrame shape: {df.shape}")
print(f"\nDataFrame columns name:\n{df.columns.tolist()}")
print(f"\nDataFrame head:\n{df.head()}")