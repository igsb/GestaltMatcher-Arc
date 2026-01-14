## gestalt_matcher_dataset.py
# GestaltMatcherDB with only basic augmentation:
# flipping, color jittering
import copy
import os
from unittest.mock import inplace

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.utils_functions import normalize, resize_with_ratio_squared, shrink_zoom_augment


class GestaltMatcherDataset(Dataset):
    def __init__(self,
                 imgs_dir,
                 target_file_path,
                 in_channels=1,
                 target_size=100,
                 img_postfix='',
                 augment=True,
                 lookup_table=None,
                 lookup_table_sort_idxs=None,
                 aspect_ratio=False,
                 remove_nonEuropeans=False,
                 id=-1):

        self.img_postfix = img_postfix
        self.target_size = target_size
        self.in_channels = in_channels
        self.imgs_dir = imgs_dir
        self.target_file = target_file_path
        self.augment = augment
        self.remove_nonEuropeans = remove_nonEuropeans
        self.run_id = id
        self.is_ancestry_experiment = True if self.run_id >= 0 else False

        if lookup_table:
            # validation
            self.lookup_table = lookup_table
            self.lut_sort_idxs = lookup_table_sort_idxs

            self.targets = self.handle_target_file()
        else:
            # training
            self.targets = self.handle_target_file()

            # Lookup table is being constructed in the function above, such that we don't pass on the incomplete one
            # due to the ancestry-exclusion experiments
            if not self.is_ancestry_experiment:
                self.lookup_table = np.array(self.targets["label"].value_counts().index.tolist())
                self.lut_sort_idxs = self.lookup_table.argsort()
                self.lookup_table = list(self.lookup_table[self.lut_sort_idxs])

        self.NUM_CLASSES = len(self.lookup_table)
        self.aspect_ratio = aspect_ratio

    def __len__(self):
        return len(self.targets)

    def get_lookup_table(self):
        return self.lookup_table

    def get_lookup_table_sort_idxs(self):
        return self.lut_sort_idxs

    def get_lookup_table_ethnicity(self):
        return self.lookup_table_ethnicity

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

        if self.is_ancestry_experiment:
            eth_id = self.lookup_table_ethnicity.index(self.targets.iloc[i]['ethnicity_category'])
            return img, target_id, eth_id
        # Debugging line:
        # print(f"{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg \t{bbox=}")

        return img, target_id

    def id_to_name(self, class_id):
        return self.lookup_table[class_id]

    def get_distribution(self):
        if self.augment:
            return np.array(self.targets.label.value_counts())[self.lut_sort_idxs]
        else:
            # present_disorder_idxs = [idx for idx, x in enumerate(self.lookup_table)
            #       if x not in np.array(self.targets.label.value_counts().keys())]
            present_disorder_idxs = np.array([np.where(self.lookup_table == x)[0][0]
                                              for x in np.array(self.targets.label.value_counts().keys())])
            distr = np.zeros(len(self.lookup_table))
            distr[present_disorder_idxs] = np.array(self.targets.label.value_counts())
            return distr[self.lut_sort_idxs]

    def get_distribution_ethnicity(self):
        return np.array(self.targets.ethnicity_category.value_counts())[self.lut_eth_sort_idxs]

    # - new test - using new columns in metadata
    def handle_target_file(self):
        # we use the supplied ethnicity column, but we can either use the specific category, or the general one
        to_use_ethnicity_column = 'ethnicity_sub_category'  # 'ethnicity_category'

        if isinstance(self.target_file, str):
            df = pd.read_csv(self.target_file, delimiter=',')
        else:
            df = self.target_file
            # we need to rename and drop some columns to retro-fit
            df['label'] = df.internal_syndrome_id
            df['subject'] = df.patient_id
            df = df[['image_id', 'subject', 'label']]

        ## in case you would like to ignore some syndromes:
        # df = df[df.label != <synd_id>]

        ### GMDB paper ethnicity exclusion ###
        if self.is_ancestry_experiment:
            remove_nonEuropeans = self.remove_nonEuropeans
            print(f"Experiment {'A' if remove_nonEuropeans else 'B3'}")
            print(f"{'Removing' if remove_nonEuropeans else 'Keeping'} non-Europeans in dataset.")

            # Load metadata
            base_dir = os.path.join(os.getcwd(), '..', 'data', 'GestaltMatcherDB', 'v1.1.0', 'gmdb_metadata')
            meta_df = pd.read_csv(f"{base_dir}/"
                                  "image_metadata_v1.1.0.tsv", usecols=['image_id', 'patient_id',
                                                                        'ethnicity_category', 'ethnicity_sub_category'],
                                  delimiter='\t')

            # remove all cases without a given ethnicity
            meta_df = meta_df.dropna(subset=['ethnicity_category'])

            # set sub-category to category if unavailable
            meta_df['ethnicity_sub_category'] = meta_df['ethnicity_sub_category'].fillna(meta_df['ethnicity_category'])
            meta_df = meta_df[meta_df.ethnicity_sub_category != 'Unknown']

            # save the ethnicity into df
            df = df.merge(meta_df[['image_id', to_use_ethnicity_column]], how='left', on='image_id')
            df = df.rename(columns={'ethnicity_sub_category': 'ethnicity_category'})

            # keep only the training images with some ethnicity
            df = df[df.image_id.isin(meta_df.image_id)]

            # keep a copy of the 'original' df before pruning some cases
            original_df = copy.deepcopy(df)

            # Filter dataset to obtain EU only or EU+Other
            # Note: Even when keeping all non-Europeans in the training set, we want the same lookup table as when we don't,
            # so we need to make the lookup table here as well. (for consistent testing/comparison)
            if self.augment:
                # Get distributions
                # - European
                df_eu = df[df.ethnicity_category == 'European']
                valid_labels = df_eu.label.value_counts().keys()[df_eu.label.value_counts().values > 6]
                df_eu = df_eu[df_eu.label.isin(valid_labels)]

                # - other
                df_other = df[df.ethnicity_category != 'European']
                df_other = df_other[df_other.label.isin(valid_labels)]

                df_other_final = None

                # first pass: randomly sample till max. same distribution per disorder
                for synd_label in valid_labels:
                    other_freq = sum(df_other.label == synd_label)
                    eu_freq = sum(df_eu.label == synd_label)
                    # to_sample = eu_freq - other_freq  # ORIGINAL
                    MIN_EUROPEAN = int(min(eu_freq, max(1, other_freq * 0.25)))  # min because we need to have enough EU to sample from; max because we want to be close to 1/4 ratio EU in Other
                    to_sample = eu_freq - other_freq - MIN_EUROPEAN

                    if to_sample > 0:  # More EU than Other -> use all Other + some EU for equal frequency
                        samples_eu = df_eu[df_eu.label == synd_label].sample(to_sample + MIN_EUROPEAN, replace=False)
                        samples_other = df_other[df_other.label == synd_label]
                    elif to_sample == 0:  # Equal EU and Other -> use all Other for equal frequency
                        samples_other = df_other[df_other.label == synd_label]
                        samples_eu = df_eu[df_eu.label == synd_label].sample(MIN_EUROPEAN, replace=False)
                    else:  # More Other than EU -> we have to subset Other for equal frequency
                        to_sample = other_freq + to_sample
                        samples_other = df_other[df_other.label == synd_label].sample(to_sample, replace=False)
                        samples_eu = df_eu[df_eu.label == synd_label].sample(MIN_EUROPEAN, replace=False)

                    df_other_final = pd.concat([df_other_final, samples_eu, samples_other])

                df = df_other_final

                # remove all non-Europeans
                if remove_nonEuropeans:
                    # df = df[df.ethnicity_category == 'European']
                    df = df_eu

            # get valid labels (depending on training/validation set); i.e., labels with enough data
            valid_labels = df.label.value_counts().keys()[df.label.value_counts().values > (6 if self.augment else 2)]

            if self.augment:
                # remove cases with a disorder less frequent than 6 occurrences - for training
                df = df[df.label.isin(valid_labels)]

                ## Experiment C: just EU (without other and 'extra' EU)
                # df = df[df.ethnicity_category == 'European']

                # create lookup table for disorders
                temp_lookup_table = np.array(df["label"].value_counts().index.tolist())
                self.lut_sort_idxs = temp_lookup_table.argsort()
                self.lookup_table = list(temp_lookup_table[self.lut_sort_idxs])

            else:
                # remove cases with a disorder not in training set - for validation
                df = df[df.label.isin(self.lookup_table)]

            self.lookup_table_ethnicity = np.array(df["ethnicity_category"].value_counts().index.tolist())
            self.lut_eth_sort_idxs = self.lookup_table_ethnicity.argsort()
            self.lookup_table_ethnicity = list(self.lookup_table_ethnicity[self.lut_eth_sort_idxs])

            print(df.ethnicity_category.value_counts())

            # we only want to save the distribution of the training set
            if self.augment:
                freq_table = pd.crosstab(df['label'], df['ethnicity_category'])
                synd_names = pd.read_csv("../data/GestaltMatcherDB/v1.1.0/gmdb_metadata/gmdb_syndromes_v1.1.0.tsv",
                                         delimiter='\t', usecols=['syndrome_id', 'syndrome_name'])
                synd_names['label'] = synd_names['syndrome_id']
                freq_table = freq_table.merge(synd_names, on='label', how='left')
                ethnicity_cols = [col for col in freq_table.columns if col not in ['label', 'syndrome_name', 'syndrome_id']]
                freq_table = freq_table[['label', 'syndrome_name'] + ethnicity_cols]
                os.makedirs("data_distributions", exist_ok=True)
                freq_table.to_csv(f"data_distributions/freq_table_EU_{'EU' if remove_nonEuropeans else 'Other'}_{self.run_id}.csv", index=False)

        ## Save split to csv-file
        # df = df[['image_id', 'subject', 'label', 'ethnicity_category']]
        # df.to_csv(f"gmdb_v1.1.0_eth_freq_{'train' if self.augment else 'val'}.csv", index=False)

        return df

    def get_num_classes(self):
        return self.NUM_CLASSES
