import mapillary.interface as mly
import requests
import json
import os

# Your token here!
# To get one, visit https://www.mapillary.com/dashboard/developer, go to 'developers',
# Then 'register application', register a new application (read access atleast),
# then copy & paste the 'Client Token' here

# TODO:: insert token.
MLY_ACCESS_TOKEN = ''
print(mly.set_access_token(MLY_ACCESS_TOKEN))


def extract_images(features, output_dir):
    """
        Extract images from given features dict using mapillary api.

        :param features: features dict, contains images information.
        :param output_dir: the directory to extract the images into.
    """
    num_images = 0

    for feature in features:
        if feature['properties']['is_pano']:
            continue

        image_id = feature['properties']['id']

        # Build the URL for downloading the image (update if needed based on your API's endpoint)
        image_url = f"https://graph.mapillary.com/{image_id}?fields=thumb_2048_url"

        # Make a request to get the image URL
        header = {'Authorization': 'OAuth {}'.format(MLY_ACCESS_TOKEN)}

        response = requests.get(image_url, headers=header)

        if response.status_code == 200:
            # Get the actual image URL from the response.
            image_data = response.json()
            if 'thumb_2048_url' in image_data:
                image_download_url = image_data['thumb_2048_url']
            else:
                print("Key 'thumb_2048_url' does not exist.")
                continue

            # Download the image.
            img_response = requests.get(image_download_url, stream=True)

            if img_response.status_code == 200:
                image_path = os.path.join(output_dir, f"{image_id}.jpg")

                with open(image_path, 'wb') as f:
                    f.write(img_response.content)
                if num_images % 25 == 0:
                    print(f"Saved image {image_id}.jpg")

                num_images += 1
            else:
                print(f"Failed to download image {image_id}")
        else:
            print(f"Failed to get image URL for ID: {image_id}")

        if num_images > 400:
            break

    print(f"Saved {num_images}")


if __name__ == "__main__":

    labels = [25, 26, 32, 33, 34, 36, 37, 20]

    for idx, label in enumerate(labels):
        json_file_path = os.path.join('./dataset_image_json', f'{label}.json')
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            print(f"loaded data for label {label}")

            features = data['features']
            if len(features) > 0:
                print(f"number of images: {len(features)}")
                output_dir = f'./prob_dataset/{label}'

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                extract_images(features, output_dir)
                print(f"images extracted from label {label}")

        except FileNotFoundError:
            print(f"The file '{json_file_path}' does not exist.")

