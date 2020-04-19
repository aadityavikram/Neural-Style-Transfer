import os
import imageio
from PIL import Image
from skimage import color
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda, ToPILImage


def load_data(input_img=None, size=512):
    transform = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        Lambda(lambda x: x[None]),
    ])

    img = Image.open(input_img)
    if img.mode == 'RGBA':
        img = color.rgba2rgb(img)
        img = Image.fromarray(img.astype('uint8'))

    img = transform(img)
    return img


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def save_image(img=None, name=""):
    transform = Compose([
        Lambda(lambda x: x[0]),
        Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
        Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
        Lambda(rescale),
        ToPILImage(),
    ])
    img = transform(img.detach().cpu())
    img.save(name)


def create_gif(path='result/output'):
    images = []
    for files in sorted(os.listdir(path), key=len):
        images.append(imageio.imread(os.path.join(path, files)))
    imageio.mimsave('result/progress.gif', images, fps=10)


def combined_result(img=None, content_img_path="", style_img_path=""):
    transform = Compose([
        Lambda(lambda x: x[0]),
        Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
        Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
        Lambda(rescale),
        ToPILImage(),
    ])
    img = transform(img.detach().cpu())
    combined = Image.new("RGB", (512 * 3, 512))
    x_offset = 0
    for image in [content_img_path, style_img_path]:
        combined.paste(Image.open(image).resize((512, 512)), (x_offset, 0))
        x_offset += 512
    combined.paste(img, (x_offset, 0))
    combined.save('result/final_output.jpg')
