from convert_panorama import panorama_to_plane

angles = [0, 180, 240, 300]

images = ["image1.jpeg", "image2.jpeg"]

i = 1
for image in images:
    for angle in angles:
        output_image = panorama_to_plane(image, 120, (256, 256), angle, 90)
        output_image.save(f'image{i}_{angle}.png')
    i += 1

