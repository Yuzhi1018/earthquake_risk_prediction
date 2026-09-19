import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import HeatMap
import ast

# Load the earthquake data
df = pd.read_csv('earthquake_data.csv')

# Display basic information about the dataset
print(f"DataFrame head:\n{df.head()}")
print(f"DataFrame info:\n{df.info()}")
print(f"DataFrame description:\n{df.describe()}")

#check for missing values
print(f"Missing values in each column:\n{df.isnull().sum()}")

#check for duplicates
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Convert time from milliseconds to datetime
df['properties.time'] = pd.to_datetime(df['properties.time'], unit='ms')

# Extract year and month for analysis
df['year'] = df['properties.time'].dt.year
df['month'] = df['properties.time'].dt.month

#drop insignificant columns
df.drop(columns=['properties.tz', 'properties.url', 'properties.detail', 'properties.felt', 'properties.net', 'properties.code', 'properties.ids', 'properties.sources', 'properties.types', 'properties.nst', 'properties.dmin', 'properties.rms', 'properties.gap'], inplace=True)

#new index- Depth
df["geometry.coordinates"] = df["geometry.coordinates"].apply(
    lambda s: ast.literal_eval(s) if isinstance(s, str) else s
)

df["depth"] = df["geometry.coordinates"].apply(
    lambda x: x[2] if isinstance(x, list) and len(x) > 2 else np.nan
)

df["depth"] = df["depth"].fillna(df["depth"].median())
                                     
# Heuristic risk score based on normalized magnitude and depth
df = df.dropna(subset=['properties.mag','depth'])
df['mag_norm'] = (df['properties.mag'] - df['properties.mag'].min()) / (df['properties.mag'].max() - df['properties.mag'].min())
df['depth_norm'] = 1- (df['depth'] - df['depth'].min()) / (df['depth'].max() - df['depth'].min())
df['tsunami_flag'] = df['properties.tsunami'].fillna(0)
#heuristic risk score calculation
df['risk_score'] = 0.7 * df['mag_norm'] + 0.3 * df['depth_norm'] 

x = df[['properties.mag', 'depth', 'tsunami_flag']]
y = df['properties.sig']

model = LinearRegression()
model.fit(x, y)

print(model.coef_)

# Plot the distribution of earthquake magnitudes
plt.figure(figsize=(10, 6))
sns.histplot(df['properties.mag'], bins=30, kde=True)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.show()

# Plot the number of earthquakes per year
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=df)
plt.title('Number of Earthquakes per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot the number of earthquakes per month
plt.figure(figsize=(10, 6))
sns.countplot(x='month', data=df)
plt.title('Number of Earthquakes per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Create HeatMap of earthquake locations
df['latitude'] = df['geometry.coordinates'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else np.nan)
df['longitude'] = df['geometry.coordinates'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
df = df.dropna(subset=['latitude', 'longitude'])

heat_data = df[['latitude', 'longitude', 'risk_score']].values.tolist()
m = folium.Map(location=[0,0], zoom_start=2)
HeatMap(heat_data, radius=18, blur=15, max_zoom=6).add_to(m)

m.save('earthquake_heatmap.html')
print("Heatmap saved as earthquake_heatmap.html")

#check csv change accordingly
df.to_csv('earthquake_data_eda.csv', index=False, encoding='utf-8-sig')