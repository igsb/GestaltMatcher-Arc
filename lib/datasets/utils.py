import os
import pandas as pd
import numpy as np

from skimage import io
from sklearn.model_selection import train_test_split

from lib.datasets.gestalt_matcher_dataset import GestaltMatcherDataset


# Function that returns the Dataset-objects for both the training set and validation set
# based on a set of parameters
def get_train_and_val_datasets(
        dataset,
        dataset_type,
        version,
        img_size,
        color_channels,
        base_dir,
        lookup_table=None,
        aspect_ratio=False,
        img_postfix='_crop_square',
        remove_nonEuropeans=False,
        data_seed=11):

    manual_split = True if data_seed >= 0 else False

    if dataset == 'gmdb':
        target_train, target_val = (None, None)
        if manual_split:
            target_train, target_val = manual_split_train_val(
                os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_metadata", f"image_metadata_{version}.tsv"),
                val_ratio=0.1, seed=data_seed)

        dataset_train = GestaltMatcherDataset(
            in_channels=color_channels,
            img_postfix=img_postfix,
            target_size=img_size,
            imgs_dir=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_crops"),
            target_file_path=target_train
                if manual_split
                else os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_metadata",
                              f"gmdb_train_images_{version}.csv"),
            lookup_table=lookup_table,
            aspect_ratio=aspect_ratio,
            remove_nonEuropeans=remove_nonEuropeans,
            id=data_seed)

        dataset_val = GestaltMatcherDataset(
            in_channels=color_channels,
            img_postfix=img_postfix,
            target_size=img_size,
            augment=False,
            imgs_dir=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_crops"),
            target_file_path=target_val
                if manual_split
                else os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_metadata",
                                          f"gmdb_val_images_{version}.csv"),
            lookup_table=(lookup_table if lookup_table else dataset_train.get_lookup_table()),
            lookup_table_sort_idxs=(None if lookup_table else dataset_train.get_lookup_table_sort_idxs()),
            aspect_ratio=aspect_ratio,
            remove_nonEuropeans=remove_nonEuropeans,
            id=data_seed)

        # loading the lookup table from a file might conflict with the sorting of the lookup table
        if lookup_table:
            print("Warning: Loading the lookup table from a file might conflict with the sorting and shown stats.")

    # Unsupported dataset (or typo)
    else:
        print(f"Dataset: {dataset} unknown, exiting.")
        exit()

    print(f"Loaded dataset: {dataset}{'-' if dataset_type else ''}{dataset_type} (version {version}) with image size "
          f"{img_size}x{img_size} in {'gray' if color_channels == 1 else 'RGB'}, while {'' if aspect_ratio else 'not'} "
          f"retaining the aspect ratio.")
    return dataset_train, dataset_val


# Instead of using predefined GMDB training and validation splits we will generate them.
# Splitting is done by patient_id and over 'frequent' disorders (where k=>7)
# k = min. patients per disorder; default = 7
# m = min. images per disorder; default = 0 (thus at least k)
def manual_split_train_val(metadata_file_path, val_ratio=0.1, k=7, m=0, seed=2):
    df = pd.read_csv(metadata_file_path, delimiter='\t')

    # Only include cases with at least k patients per disorder
    patient_counts_per_disorder = df.groupby('internal_syndrome_id')['patient_id'].nunique()
    valid_disorders = patient_counts_per_disorder[patient_counts_per_disorder >= k].index
    df = df[df['internal_syndrome_id'].isin(valid_disorders)]

    # Only include cases with at least m images per disorder
    disorder_counts = df['internal_syndrome_id'].value_counts()
    valid_disorders = disorder_counts[disorder_counts >= m].index
    df = df[df['internal_syndrome_id'].isin(valid_disorders)]

    # remove mixed ethnicity from df, include all in val set
    mixed_eth_cases = df[df.ethnicity_sub_category == 'Mixed ancestry']
    df = df[df.ethnicity_sub_category != 'Mixed ancestry']

    ## Splitting the data into training and validation sets ensuring no overlap in patient_ids
    # Grouping by disorder to ensure they are represented in both splits
    grouped = df.groupby('internal_syndrome_id')

    val_patients = []
    # seed = np.random.get_state()[1][0]
    print(f"data_seed={seed}")
    rng = np.random.default_rng(seed)
    # Sample patient_ids per disorder to match the ratio of n
    for idx, (_, group) in enumerate(grouped):
        # We can change the logic here to use a loop which continues sampling patient_id until num_images > n
        patients_in_group = group.patient_id.unique()
        val_patients.extend(rng.choice(patients_in_group, np.max((1, int(len(patients_in_group)*val_ratio))), replace=False))

    val_data = df[df.patient_id.isin(val_patients)]
    train_data = df[~df.patient_id.isin(val_patients)]

    # extend val set with mixed ethnicities
    val_data = val_data.append(mixed_eth_cases)

    return train_data, val_data
