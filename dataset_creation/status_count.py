import os
import json

lats = [40.416167, 25.7617, 50.0755, 42.3601, 41.8781, 13.7563, 33.4484, 41.3851, 51.5074, 59.9139, 48.851630, 43.64429,
        -37.8136, -34.6037, 38.7223, 19.4326, 41.888588]

lngs = [-3.707893, -80.1918, 14.4378, -71.0589, -87.6298, 100.5018, -112.074, 2.1734, -0.1278, 10.7522, 2.326903,
        -79.388404, 144.9631, -58.3816, -9.1393, -99.1332, 12.490206]

labels = [14, 56, 43, 13, 2, 77, 59, 76, 109, 72, 41, 85, 54, 71, 78, 45, 84]

count = 0

for idx, label in enumerate(labels):
    json_file_path = os.path.join('./dataset_image_json', f'{label}.json')
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        print(f"for label: {label} at {lats[idx]}, {lngs[idx]} number of images: {len(data['features'])}")
        count += 1

    except FileNotFoundError:
        print(f"The file '{json_file_path}' does not exist.")

print(len(labels))
print(count)
