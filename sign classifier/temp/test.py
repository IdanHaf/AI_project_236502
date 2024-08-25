import numpy as np
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision.transforms import transforms

transform = transforms.Compose([

    transforms.Resize((800, 800)),
])

img = Image.open('1.jpg')
js = json.loads(open('1.json').read())
width_ratio = 800 / js['width']
height_ratio = 800 / js['height']
img = transform(img)
print(js)
img_draw = ImageDraw.Draw(img)
color = 'red'
alpha = 125


rects = Image.new('RGBA', img.size)
rects_draw = ImageDraw.Draw(rects)
for obj in js['objects']:
    x1 = obj['bbox']['xmin']
    y1 = obj['bbox']['ymin']
    x2 = obj['bbox']['xmax']
    y2 = obj['bbox']['ymax']
    x1 *= width_ratio
    x2 *= width_ratio
    y1 *= height_ratio
    y2 *= height_ratio

    print(f'x1 {x1} y1 {y1} x2 {x2} y2 {y2}')
    color_tuple = ImageColor.getrgb(color)
    if len(color_tuple) == 3:
        color_tuple = color_tuple + (alpha,)
    else:
        color_tuple[-1] = alpha

    rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1), fill=color_tuple)
    img_draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='black', width=1)

    class_name = obj['label']
    img_draw.text((x1 + 5, y1 + 5), class_name)
img.show()
