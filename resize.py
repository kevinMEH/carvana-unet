# Resizes the images in the datasets to prevent excessive data usage

from PIL import Image
from os import listdir, path

image_directory = "./data/train_masks"
resized_directory = "./data/train_small_masks"
image_names = []

for file_name in listdir(image_directory):
    if path.isfile(path.join(image_directory, file_name)) and not file_name.startswith("."):
        image_names.append(file_name)

for image_name in image_names:
    image_path = path.join(image_directory, image_name)
    image = Image.open(image_path)
    resized = image.resize((image.size[0]//2, image.size[1]//2), resample=Image.Resampling.NEAREST)
    resized.save(path.join(resized_directory, image_name))