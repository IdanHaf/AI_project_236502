import os
import json

from tqdm import tqdm

dir_name = os.path.join('dataset', "splits")
dir_img = os.path.join('dataset', "annotations")
res = ""
label_res = {}
images = [os.path.splitext(file)[0] for file in os.listdir("images")]

for file_name in os.listdir(dir_img):
    if os.path.splitext(file_name)[0] in images:
        res = res + file_name + '\n'

with open("annotated_images.txt", 'w') as f:
    f.write(res)

images = [line.strip() for line in open('annotated_images.txt', 'r').readlines()]

for idx in tqdm(range(len(images))):
    image = images[idx]
    with open(os.path.join(dir_img, image), 'r') as f:
        image_data = json.load(f)
        for obj in image_data['objects']:
            label = obj['label']
            if label not in label_res.keys():
                label_res[label] = len(label_res.keys()) + 1

with open("labels.json", 'w') as f:
    f.write(json.dumps(label_res))
