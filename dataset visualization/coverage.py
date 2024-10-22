import folium
import pandas as pd
from folium.plugins import HeatMap


# Load data from CSV
df = pd.read_csv('images.csv')

# Create a map centered around the average location
map_center = [df['lat'].mean(), df['lng'].mean()]
mymap = folium.Map(location=map_center)

heat_data = df[['lat', 'lng']].values.tolist()
HeatMap(heat_data, radius=10).add_to(mymap)


# Save the map to an HTML file
mymap.show_in_browser()
mymap.save("map.html")