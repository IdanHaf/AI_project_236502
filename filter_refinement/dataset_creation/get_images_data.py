import json
import mapillary.interface as mly
import os

# TODO:: insert token.
MLY_ACCESS_TOKEN = ''
print(mly.set_access_token(MLY_ACCESS_TOKEN))

if __name__ == "__main__":
    lats = [
        53.517986750936, 56.03028801607326, -7.12546145223412, 40.333962763049755,
        42.45960113775601, -3.892700782086692, 14.762016650776587, 61.59631770431949
    ]

    lngs = [
        -113.41862119755, 92.91863877015203, -34.85286106392556, -75.94587071350645,
        -96.37816418509948, 119.5646817408051, 104.49389685378083, -149.11065998014124
    ]

    labels = [
        20, 25, 26, 32, 33, 34, 36, 37
    ]

    visited_labels = [
        14, 56, 43, 13, 2, 77, 59, 76, 109, 72, 41, 85, 54, 71, 78,
        45, 84, 9, 51, 3, 102, 40, 35, 44, 100, 8, 53, 28, 60, 92, 19,
        10, 118, 27, 117, 15, 61, 108, 67, 94, 98, 22, 18, 62, 91, 105,
        4, 5, 1, 6, 11, 12, 16, 17, 64, 7, 21, 23, 24, 29, 30, 31, 20,
        25, 26, 32, 33, 34, 36, 37
    ]

    found_lst = []

    outdir = './dataset_image_json'
    os.makedirs(outdir, exist_ok=True)

    idx = 0

    for lat, lng in zip(lats, lngs):
        if labels[idx] in visited_labels:
            idx += 1
            print(f"already visited: {labels[idx]}")
            continue

        visited_labels.append(labels[idx])

        print("start loading data")

        data = mly.get_image_close_to(lat, lng, radius=100)
        rad = 100
        while data is None and rad <= 500000:
            rad *= 3
            data = mly.get_image_close_to(lat, lng, radius=rad)
            print(f"no data, increasing radius for {lat}, {lng} to {rad}")

        if rad > 500000:
            idx += 1
            continue

        data = data.to_dict()
        while len(data['features']) < 500 and rad < 500000:
            print(f"increasing radius for {lat}, {lng} to {rad}")
            data = mly.get_image_close_to(lat, lng, radius=rad).to_dict()
            rad *= 2

        print(f"number of images: {len(data['features'])}")
        json_file_path = os.path.join(outdir, f'{labels[idx]}.json')
        found_lst.append(labels[idx])

        with open(json_file_path, mode="w") as f:
            json.dump(data, f, indent=4)

        print("loaded data")
        idx += 1

    print(found_lst)
