import os
import random
from flask import Flask, jsonify, render_template
import folium
from Pipeline import Atlas
from PIL import Image

weight_folder = os.path.join('model_resources', 'weights')
region_model = os.path.join(weight_folder, 'reg.pth')
lang_model = os.path.join(weight_folder, 'lang.pth')
moco_model = os.path.join(weight_folder, 'moco.pth')
refine_model = os.path.join(weight_folder, 'refine.pth')
baseset_file = os.path.join('model_resources', 'baseset.csv')

pipeline = Atlas(region_model, lang_model, moco_model, refine_model, 15, baseset_file)

app = Flask(__name__)

# Folder with images inside 'static/Images'
IMAGE_FOLDER = os.path.join('static', 'Images')

# Function to pick a random image and extract lat/lon
def pick_random_image():
    files = os.listdir(IMAGE_FOLDER)
    image = random.choice(files)
    # Assuming filename is in the format 'lat-lon.jpg'
    lat, lon = map(float, image.rsplit('.', 1)[0].split(','))
    return image, lat, lon

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/random-image')
def get_random_image():
    image, lat, lon = pick_random_image()
    return jsonify({'image': image, 'lat': lat, 'lon': lon})

@app.route('/generate-map/<lat>/<lon>')
def generate_map(lat, lon):
    pred = pipeline.predict(["static\\Images\\"+str(lat) +","+str(lon) + ".jpg"])
    lat_pred = pred[0][0]
    lon_pred = pred[0][1]
    map_obj = folium.Map(location=[0, 0], zoom_start=2)
    
    if lat != 'no-pin' and lon != 'no-pin':
        lat, lon = float(lat), float(lon)
        map_obj = folium.Map(location=[lat, lon], zoom_start=2)
        folium.Marker([lat, lon], popup='Image Location', icon=folium.Icon(color='red')).add_to(map_obj)
    folium.Marker([lat_pred, lon_pred], popup='Predicted Location', icon=folium.Icon(color='blue')).add_to(map_obj)

    map_path = os.path.join('static', 'map.html')
    map_obj.save(map_path)

    return jsonify({'map': '/static/map.html'})



if __name__ == '__main__':
    app.run(port=1337)
