import os
import json

from tqdm import tqdm

dir_name = "dataset\splits"
dir_img = "dataset\\annotations"
res = ""
label_res = {}

for file_name in os.listdir(dir_img):
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

with open("labels.txt", 'w') as f:
    f.write(json.dumps(label_res))
