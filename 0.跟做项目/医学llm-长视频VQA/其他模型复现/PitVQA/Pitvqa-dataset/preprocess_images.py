import os
import glob
from PIL import Image, ImageOps
from pathlib import Path

'''
Before run this code, your folder system should look like this:
project-name/
├── images/
│   ├── video_01/
│   └── video_02/
│   └── ...
│   └── video_25/
├── preprocessed_images/
├── preprocess_images.py
note: 'preprocessed_images' is an empty folder before running this code.
'''

if __name__ == '__main__':

    images_path = Path('./images')
    output_path = Path('./preprocessed_images')
    video_sequence = [f"video_{i:02}" for i in range(1, 26)]

    for seq in video_sequence:
        folder_path = images_path / seq
        images = folder_path.glob('*.png')

        # create a new folder (e.g. video_01/) in the 'preprocessed_images' folder
        new_folder = output_path / seq
        new_folder.mkdir(parents=True, exist_ok=True)

        for img in images:
            image = Image.open(img)

            # process images
            crop_box = (295, 50, 935, 690)
            cropped_image = image.crop(crop_box)  # crop images
            resized_image = cropped_image.resize((224, 224), Image.BICUBIC)  # resize images

            # save processed images
            save_path = new_folder / img.name
            resized_image.save(save_path)
        print(seq + ' done.')
