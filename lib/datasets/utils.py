import os
from skimage import io

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
        img_postfix='_crop_square'):

    if dataset == 'gmdb':
        dataset_train = GestaltMatcherDataset(
            in_channels=color_channels,
            img_postfix=img_postfix,
            target_size=img_size,
            imgs_dir=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_align"),
            target_file_path=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_metadata",
                                          f"gmdb_train_images_{version}.csv"),
            lookup_table=lookup_table,
            aspect_ratio=aspect_ratio)

        dataset_val = GestaltMatcherDataset(
            in_channels=color_channels,
            img_postfix=img_postfix,
            target_size=img_size,
            augment=False,
            imgs_dir=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_align"),
            target_file_path=os.path.join(base_dir, "GestaltMatcherDB", version, "gmdb_metadata",
                                          f"gmdb_val_images_{version}.csv"),
            lookup_table=(lookup_table if lookup_table else dataset_train.get_lookup_table()),
            aspect_ratio=aspect_ratio)

    # Unsupported dataset (or typo)
    else:
        print(f"Dataset: {dataset} unknown, exiting.")
        exit()

    print(f"Loaded dataset: {dataset}{'-' if dataset_type else ''}{dataset_type} (version {version}) with image size "
          f"{img_size}x{img_size} in {'gray' if color_channels == 1 else 'RGB'}, while {'' if aspect_ratio else 'not'} "
          f"retaining the aspect ratio.")
    return dataset_train, dataset_val

