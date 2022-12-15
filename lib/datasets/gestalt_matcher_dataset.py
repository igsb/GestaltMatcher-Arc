## gestalt_matcher_dataset.py
# GestaltMatcherDB with only basic augmentation:
# flipping, color jittering

import os

import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from lib.utils import normalize, resize_with_ratio_squared, shrink_zoom_augment


class GestaltMatcherDataset(Dataset):
    def __init__(self,
                 imgs_dir,
                 target_file_path,
                 in_channels=1,
                 target_size=100,
                 img_postfix='',
                 augment=True,
                 lookup_table=None,
                 aspect_ratio=False):

        self.img_postfix = img_postfix
        self.target_size = target_size
        self.in_channels = in_channels
        self.imgs_dir = imgs_dir
        self.target_file = target_file_path

        self.targets = self.handle_target_file()

        if lookup_table:
            self.lookup_table = lookup_table
        else:
            self.lookup_table = self.targets["label"].value_counts().index.tolist()
            self.lookup_table.sort()

        self.augment = augment
        self.NUM_CLASSES = len(self.lookup_table)
        self.aspect_ratio = aspect_ratio

    def __len__(self):
        return len(self.targets)

    def get_lookup_table(self):
        return self.lookup_table

    def preprocess(self, img):

        # # Randomly shrink the img in range 50 < x < 150 and resize to target size afterwards
        # # where x is the longest dimension size, the shortest size will be scaled according to ratio
        if self.augment:
            img = shrink_zoom_augment(img, min_size=[50, 100], aspect_ratio=False, p=0.1)  # randomly select

        # Resize the image retaining the original image ratio and padding size with black pixels to square the image
        if self.aspect_ratio:
            img = resize_with_ratio_squared(img, self.target_size)
        else:
            img = A.resize(img, self.target_size, self.target_size)

        if self.augment:
            flip_jitter_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(hue=0.1, always_apply=True)
            ])
            img = flip_jitter_aug(image=img)["image"]

        # desired number of channels is 1, so we convert to gray,
        # if num_channels = 3 we will randomly convert to 3-channel gray (as augmentation)
        if self.in_channels == 1:
            img = A.to_gray(img)[:, :, 0]
        else:
            # TODO: Decide on a probability to convert to gray; maybe equal to the ratio of gray images in the dataset?
            img = A.ToGray(p=(0.1 if self.augment else 0.))(image=img)["image"]

        img = ToTensorV2()(image=img)["image"]
        return normalize(img, type='arcface')

    def __getitem__(self, i, to_augment=True):
        img = cv2.imread(os.path.join(self.imgs_dir, f"{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg"))
        target_id = self.lookup_table.index(self.targets.iloc[i]['label'])

        img = self.preprocess(img)

        # Debugging line:
        # print(f"{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg \t{bbox=}")

        return img, target_id

    def id_to_name(self, class_id):
        return self.lookup_table[class_id]

    def get_distribution(self):
        return list(self.targets.label.value_counts())

    def handle_target_file(self):
        df = pd.read_csv(self.target_file, delimiter=',')

        ## in case you would like to ignore some syndromes:
        # df = df[df.label != <synd_id>]

        return df

    def get_num_classes(self):
        return self.NUM_CLASSES
