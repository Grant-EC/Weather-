#This was included code from where I originally found the dataset

# ==============================
# Weather Data Analysis Script
# ==============================

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# 1. Load and inspect data
# ==============================

# Use raw string or forward slashes to avoid Unicode errors
data_path = r'C:\Users\GECross\Downloads\seattle-weather.csv' #specific for where it is saved in my device
df = pd.read_csv(data_path)

print("\n--- Data Overview ---")
print(df.info())
print("\nMissing values:\n", df.isna().sum())

# ==============================
# 2. Clean and preprocess data
# ==============================

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Drop rows with missing values
df.dropna(inplace=True)

# Sort by date (in place)
df.sort_values(by="date", inplace=True)

# ==============================
# 3. Exploratory Data Analysis
# ==============================

# Numerical and categorical summary
print("\n--- Numerical Summary ---")
print(df.describe())
print("\n--- Categorical Summary ---")
print(df.describe(include='object'))

# ==============================
# 4. Matplotlib & Seaborn Plots
# ==============================

# Scatter plot: precipitation vs temperature
plt.figure(figsize=(6,4))
plt.scatter(df["precipitation"], df["temp_max"], color="red", alpha=0.6)
plt.xlabel("Precipitation")
plt.ylabel("Max Temperature")
plt.title("Precipitation vs Max Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

# Pie chart: weather frequency
weather_counts = df['weather'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Weather Distribution")
plt.tight_layout()
plt.show()

# Heatmap of top 10 weathers (averages)
top_weather = df.groupby('weather')[['precipitation', 'temp_max', 'temp_min', 'wind']].mean().nlargest(10, 'temp_max')
plt.figure(figsize=(10,6))
sns.heatmap(top_weather, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Average Weather Metrics (Top 10 by Temp Max)")
plt.tight_layout()
plt.show()

# ==============================
# 5. Plotly Visualizations
# ==============================

# Interactive histogram
fig = px.histogram(df, x="wind", template='plotly', title='Wind Distribution',
                   text_auto=True, opacity=0.8)
fig.update_layout(bargap=0.3)
fig.show()

# Interactive pie chart (top 10 weather types by temp_max)
top_weather_pie = top_weather.reset_index()
fig = go.Figure(data=[go.Pie(
    labels=top_weather_pie['weather'],
    values=top_weather_pie['temp_max'],
    textinfo='label+percent',
    hole=0.3,
    pull=[0.1 if i == 0 else 0 for i in range(len(top_weather_pie))]
)])
fig.update_layout(title_text='Top 10 Weather Types by Max Temperature')
fig.show()
