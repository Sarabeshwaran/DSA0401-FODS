import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the customer reviews data
data = pd.read_csv("customer_reviews.csv")

# Display the first few rows of the data to understand its structure
print(data.head())

# Calculate the average rating
average_rating = data['rating'].mean()
print("Average Rating:", average_rating)

# Calculate the standard error of the mean
standard_error = data['rating'].std() / np.sqrt(len(data))

# Define the desired confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the z-score for the desired confidence level
z_score = norm.ppf((1 + confidence_level) / 2)

# Calculate the margin of error
margin_of_error = z_score * standard_error

# Calculate the confidence interval
confidence_interval_lower = average_rating - margin_of_error
confidence_interval_upper = average_rating + margin_of_error

print("Confidence Interval:", (confidence_interval_lower, confidence_interval_upper))
