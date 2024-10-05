import json
import mapillary.interface as mly
import os

# TODO:: add token.
MLY_ACCESS_TOKEN = ''
print(mly.set_access_token(MLY_ACCESS_TOKEN))

if __name__ == "__main__":
    # lats = [35.659659, 41.878498, 50.846004]
    # lngs = [139.700627, -87.625744, 4.349758]
    # labels = [51, 2, 3]

    lats = []

    lngs = []

    labels = []

    visited_labels = [14, 56, 43, 13, 2, 77, 59, 76, 109, 72, 41, 85, 54, 71, 78, 45, 84, 9]

    outdir = './dataset_image_json'
    os.makedirs(outdir, exist_ok=True)

    idx = 0

    for lat, lng in zip(lats, lngs):
        if labels[idx] in visited_labels:
            idx += 1
            print(f"already visited: {labels[idx]}")
            continue

        print("start loading data")

        data = mly.get_image_close_to(lat, lng, radius=100)
        if data is None:
            print(f"no data found for {lat}, {lng}")
            idx += 1
            continue

        data = data.to_dict()
        rad = 150
        while len(data['features']) < 500 and rad < 3000:
            print(f"increasing radius for {lat}, {lng} to {rad}")
            data = mly.get_image_close_to(lat, lng, radius=rad).to_dict()
            rad += 100

        print(f"number of images: {len(data['features'])}")
        json_file_path = os.path.join(outdir, f'{labels[idx]}.json')

        with open(json_file_path, mode="w") as f:
            json.dump(data, f, indent=4)

        print("loaded data")
        idx += 1
