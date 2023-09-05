import pandas as pd
import numpy as np

data = pd.read_csv('temperature_data.csv')

mean_temperatures = data.groupby('City')['Temperature'].mean()

std_dev_temperatures = data.groupby('City')['Temperature'].std()

temperature_range = data.groupby('City')['Temperature'].max() - data.groupby('City')['Temperature'].min()
city_with_highest_range = temperature_range.idxmax()

city_with_lowest_std_dev = std_dev_temperatures.idxmin()


print("Task 1: Mean Temperature for Each City")
print(mean_temperatures)

print("\nTask 2: Standard Deviation of Temperature for Each City")
print(std_dev_temperatures)

print("\nTask 3: City with the Highest Temperature Range")
print(f"City: {city_with_highest_range}, Temperature Range: {temperature_range.max()}")

print("\nTask 4: City with the Most Consistent Temperature (Lowest Standard Deviation)")
print(f"City: {city_with_lowest_std_dev}, Standard Deviation: {std_dev_temperatures.min()}")
