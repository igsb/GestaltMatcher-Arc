import random
import base64
import torch
import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt


# Function that helps set each work's seed differently (consistently)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Function to plot an image from a tensor or np array
# optionally waits for button press before continuing
def imshow(img, wait=False):
    plt.close()

    if torch.is_tensor(img):
        img_t = img.permute(1, 2, 0)
    elif type(img).__module__ == np.__name__:
        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_t)
    if wait:
        plt.waitforbuttonpress()


# Function to normalize an image's pixel values
# Either in range [0,1], [0,255], or [-1,1]
def normalize(img, type='float'):
    if type == 'int':
        normalized = (img - img.min()) / (img.max() - img.min())
        return (normalized * 255).int()

    elif type == 'arcface':
        return (img.float() / 255 - 0.5) / 0.5

    else: #type == 'float':
        return (img - img.min()) / (img.max() - img.min())


# Function that shrinks an image to min_size to later introduce some up-scaling
# artifacts (to address the small images in the dataset)
# min_size is a list/tuple of len() =2, randomly select the min_size in that range
# interpolation options: cv2.INTER_LINEAR, cv2.INTER_CUBIC
def shrink_zoom_augment(img, min_size=75, aspect_ratio=True, interpolation=cv2.INTER_LINEAR, p=0.1):
    # Only shrink if p-case is met
    img_p = img
    if np.random.rand(1) < p:
        # If the min_size is of len(2), randomly select int in that range:
        if isinstance(min_size, (list, tuple)):
            if len(min_size) == 2:
                min_size = int(np.random.rand(1)*(min_size[1]-min_size[0])+min_size[0])

        if aspect_ratio:
            img_p = resize_with_ratio(img_p, min_size, interpolation=interpolation)
        else:
            img_p = A.resize(img, min_size, min_size, interpolation=interpolation)

    return img_p


# Resizes the input image retaining the aspect ratio
def resize_with_ratio(img, max_dim_size, interpolation=cv2.INTER_LINEAR):
    aug = A.LongestMaxSize(max_size=max_dim_size, interpolation=interpolation)
    img_p = aug(image=img)["image"]
    return img_p


# Resizes the input image toa squared image, retaining the original aspect ratio through zero padding
def resize_with_ratio_squared(img, max_dim_size, pad_color=0, interpolation=cv2.INTER_LINEAR):
    transform = A.Compose([
        A.LongestMaxSize(max_size=max_dim_size, interpolation=interpolation),
        A.PadIfNeeded(
            min_height=max_dim_size,
            min_width=max_dim_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(pad_color, pad_color, pad_color)
        )
    ])
    padded = transform(image=img)["image"]

    return padded


# Crop the image to a squared image where the bbox is in the center
# if the bbox cannot be in the center because of a lack of pixels on any size, we pad it with the pad_color
def resize_crop_with_ratio_squared(img, bbox, max_dim_size, pad_color=0, interpolation=cv2.INTER_LINEAR):
    w,h = [bbox[2]-bbox[0], bbox[3]-bbox[1]]  # (width, height)

    if w >= h:
        # we should pad the height
        padding = w-h
        new_bbox = [bbox[0], bbox[1]-padding//2, bbox[2], bbox[3]+padding//2]

        h_top_pad, h_bottom_pad = 0, 0
        # does the padded bbox exceed image boundaries?
        if new_bbox[1] < 0:
            # zero pad the top-side
            h_top_pad = abs(new_bbox[1])

        if new_bbox[3] > img.shape[0]:
            # zero pad the bottom-side
            h_bottom_pad = new_bbox[3] - img.shape[0]

        img = A.pad_with_params(img, h_top_pad, h_bottom_pad, 0, 0, border_mode=cv2.BORDER_CONSTANT, value=pad_color)
        new_bbox = [new_bbox[0], new_bbox[1]+h_top_pad, new_bbox[2], new_bbox[3]+h_top_pad]

    else:
        # we should pad the width
        padding = h-w
        new_bbox = [bbox[0]-padding//2, bbox[1], bbox[2]+padding//2, bbox[3]]
        w_left_pad, w_right_pad = 0, 0
        # does the padded bbox exceed image boundaries?
        if new_bbox[0] < 0:
            # zero pad the left-side
            w_left_pad = abs(new_bbox[0])

        if new_bbox[2] > img.shape[1]:
            # zero pad the right-side
            w_right_pad = new_bbox[2] - img.shape[1]

        img = A.pad_with_params(img, 0, 0, w_left_pad, w_right_pad, border_mode=cv2.BORDER_CONSTANT, value=pad_color)
        new_bbox = [new_bbox[0]+w_left_pad, new_bbox[1], new_bbox[2]+w_left_pad, new_bbox[3]]

    img = A.crop(img, *new_bbox)
    img = A.resize(img, max_dim_size, max_dim_size, interpolation=interpolation)

    return img


def readb64(uri):
    # encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encodeb64(uri):
    # encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img