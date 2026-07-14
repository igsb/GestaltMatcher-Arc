"""Target-vs-target pairwise distance matrix and target-vs-gallery rank matrix.

Ported from notebook cells 12 and 13.
"""

import numpy as np
import pandas as pd

from lib.cohort_analysis import embeddings


def compute_distance_matrix(target_representation_df, target_image_ids, config):
    """Target-vs-target mean cosine distance matrix (notebook cell 12).

    Returns (distance_df, target_tta). target_tta is returned because the rank
    matrix needs the same TTA arrays and the notebook reuses them directly.
    """
    target_representation = embeddings.get_embeddings_for_ids(
        target_representation_df, target_image_ids
    )

    target_tta = embeddings.representation_list_to_tta_arrays(
        target_representation, n_tta=config["n_tta"]
    )

    distance_matrix = embeddings.calculate_distance_matrix(
        target_tta,
        target_tta,
        metric=config["distance_metric"],
    )

    distance_df = pd.DataFrame(
        distance_matrix,
        index=target_image_ids,
        columns=target_image_ids,
    )

    return distance_df, target_tta


def compute_rank_matrix(target_tta, gallery_representation_df, target_image_ids, config):
    """Target-vs-(gallery + target) rank matrix (notebook cell 13).

    For each target image, every other target image is ranked among the combined
    gallery + target pool by distance. Rank 0 is the nearest image (an image's
    own rank against itself is 0).

    Note the transpose: the notebook builds the DataFrame from
    np.array(target_ranks).T, so rank_df is the transpose of the row-wise ranks.
    Preserved exactly.

    TODO: n_tta is assumed to be identical for the gallery and target embeddings;
    the TTA slices are combined index-wise with no validation. A mismatch would
    silently mispair slices rather than fail. Not changed in this pass.
    """
    gallery_image_ids_loaded = gallery_representation_df["img_name"].astype(str).values

    gallery_representation = embeddings.get_embeddings_for_ids(
        gallery_representation_df, gallery_image_ids_loaded
    )

    gallery_tta = embeddings.representation_list_to_tta_arrays(
        gallery_representation, n_tta=config["n_tta"]
    )

    combined_gallery_tta = [
        np.append(gallery_tta[i], target_tta[i], axis=0)
        for i in range(config["n_tta"])
    ]
    combined_image_ids = np.append(gallery_image_ids_loaded, target_image_ids)

    all_distance = embeddings.calculate_distance_matrix(
        target_tta,
        combined_gallery_tta,
        metric=config["distance_metric"],
    )

    target_ranks = []
    for row_index, image_id in enumerate(target_image_ids):
        sorted_indices = np.argsort(all_distance[row_index])
        sorted_ids = combined_image_ids[sorted_indices]
        ranks = [np.where(sorted_ids == other_id)[0][0] for other_id in target_image_ids]
        target_ranks.append(ranks)

    rank_df = pd.DataFrame(
        np.array(target_ranks).T,
        index=target_image_ids,
        columns=target_image_ids,
    )

    return rank_df
