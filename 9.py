import pandas as pd

average_price_per_location = property_data.groupby('location')['listing_price'].mean()
print(average_price_per_location)

properties_more_than_4_bedrooms = property_data[property_data['number_of_bedrooms'] > 4]
num_properties_more_than_4_bedrooms = len(properties_more_than_4_bedrooms)
print("Number of properties with more than four bedrooms:", num_properties_more_than_4_bedrooms)

property_with_largest_area = property_data[property_data['area_sq_ft'] == property_data['area_sq_ft'].max()]
print("Property with the largest area:")
print(property_with_largest_area)
