import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from convert_panorama import panorama_to_plane
import os
from concurrent.futures import ProcessPoolExecutor

angles = [0, 180, 240, 300, 340]
images_data = pd.read_csv('images.csv')
indices = math.ceil(len(images_data) / 1024)
chunks = np.array_split(images_data, indices)
current_dir = os.getcwd()
images_dir = os.path.join(current_dir, 'images')
res_dir = os.path.join(current_dir, 'results')
data = pd.DataFrame(columns=images_data.columns)


def augment_batch(batch, im_dir, res_imdir):
    new_rows = []

    for image_idx in range(len(batch)):
        image_row = batch.iloc[image_idx]
        image_name = image_row['id']
        image_address = os.path.join(im_dir, f'{image_name}.jpeg')

        for angle_idx in range(len(angles)):
            output_image = panorama_to_plane(image_address, 120, (256, 256), angles[angle_idx], 90)
            res_address = os.path.join(res_imdir, f'{image_name}{angle_idx}.jpeg')

            new_row = {
                'id': f'{image_name}{angle_idx}',
                'lat': image_row['lat'],
                'lng': image_row['lng'],
                'date_taken': image_row['date_taken']
            }
            new_rows.append(new_row)
            output_image.save(res_address)

    return new_rows


def process_batches(batches, im_dir, res_imdir):
    all_new_rows = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(augment_batch, batch, im_dir, res_imdir)
            for idx, batch in enumerate(batches)
        ]
        for future in tqdm(futures):
            all_new_rows.extend(future.result())

    return pd.DataFrame(all_new_rows)


if __name__ == '__main__':
    augmented_data = process_batches(chunks, images_dir, res_dir)
    augmented_data.to_csv('augmented_data.csv', index=False)
