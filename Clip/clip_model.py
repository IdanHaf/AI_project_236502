import torch
import clip
from PIL import Image
import numpy as np

"""
Taken from https://github.com/openai/CLIP.
"""

image_paths = ["237835374806467.jpg", "2907789836163958.jpg", "457162172242493.jpg"]
text_labels = [
    "urban image from Asia",
    "rural landscape image from Asia",
    "urban image from USA",
    "rural landscape image from USA",
]


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Gpu is available: " + str(torch.cuda.is_available()))

model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(text_labels).to(device)

for image_path in image_paths:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", np.round(probs, 2))
