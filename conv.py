from PIL import Image
import os

path = 'Lung Disease Dataset/val/Tuberculosis'

def resample_img(img_name):
    size = 512, 512
    try:
        im = Image.open(f'{path}/{img_name}.jpeg')
    except FileNotFoundError:
        try:
            im = Image.open(f'{path}/{img_name}.jpg')
        except FileNotFoundError:
            return 0

    im = im.resize(size, Image.Resampling.LANCZOS)
    im.save(f'{path}/{img_name}.png')
    try:
        os.remove(f'{path}/{img_name}.jpeg')
    except FileNotFoundError:
        os.remove(f'{path}/{img_name}.jpg')


files = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
for i in files:
    resample_img(i)