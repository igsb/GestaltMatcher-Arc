"""Embedding lookup, TTA reshaping, and distance matrix computation.

Ported from the embedding helpers in notebook cells 7 and 25.

TODO: calculate_distance_matrix (cell 7, used by the pairwise step) and
cosine_distance_matrix_mean_tta (cell 25, used by the cross-cohort step) are two
independent implementations of the same mean cosine distance over TTA. They
should agree, but that has not been verified. Both are kept, each feeding the
step the notebook fed it, so this refactor cannot change results either way.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def representation_list_to_tta_arrays(representation_values, n_tta=None):
    """Regroup per-image TTA lists into one array per TTA slice.

    Input:  a sequence of length n_images, each element a list of n_tta vectors.
    Output: a list of length n_tta, each element an (n_images, dim) array.
    """
    # representation_values: array/list of per-image lists, where each image has TTA vectors
    if len(representation_values) == 0:
        return []

    if n_tta is None:
        n_tta = len(representation_values[0])

    return [
        np.array([representation_values[j][i] for j in range(len(representation_values))])
        for i in range(n_tta)
    ]


def calculate_distance_matrix(query_tta, gallery_tta, metric="cosine"):
    """Mean distance over TTA slices between every query and every gallery image.

    Returns an (n_query, n_gallery) matrix.
    """
    # returns shape: n_query x n_gallery, averaged over TTA
    dists = np.stack(
        [pairwise_distances(query_tta[i], gallery_tta[i], metric=metric)
         for i in range(len(query_tta))],
        axis=1,
    )

    return np.mean(dists, axis=1)


def get_embeddings_for_ids(rep_df, image_ids):
    """Look up the representation lists for image_ids, in the order given.

    Raises if any requested ID is absent from rep_df.

    TODO: this is a linear scan of rep_df per ID, i.e. O(n_ids * n_rows). It is
    left as-is on purpose so this refactor cannot change results. Index rep_df by
    img_name once reproducibility against the notebook is confirmed.
    """
    image_ids = list(map(str, image_ids))

    missing = [x for x in image_ids if x not in set(rep_df["img_name"].astype(str))]
    if missing:
        raise ValueError(
            "Missing {} image IDs in representation table. Examples: {}".format(
                len(missing), missing[:10]
            )
        )

    return [
        rep_df.loc[rep_df["img_name"].astype(str) == str(i), "representations"].values[0]
        for i in image_ids
    ]


def get_embeddings_tensor_for_ids(image_ids, rep_df):
    """Stack the representations for image_ids into an (N, T, D) tensor.

    Notebook cell 25. Accepts a representation table keyed by an img_name column,
    an image_id column, or the index. Returns (E, ordered_image_ids).
    """
    image_ids = list(map(str, image_ids))
    rep = rep_df.copy()

    possible_rep_cols = ["representations", "representation", "embedding", "embeddings"]
    rep_col = next((c for c in possible_rep_cols if c in rep.columns), None)

    if rep_col is None:
        raise ValueError(
            "Cannot find representation column. Tried {}. Available columns: {}".format(
                possible_rep_cols, rep.columns.tolist()
            )
        )

    if "img_name" in rep.columns:
        rep["img_name"] = rep["img_name"].astype(str)
        rep = rep.drop_duplicates("img_name").set_index("img_name")
    elif "image_id" in rep.columns:
        rep["image_id"] = rep["image_id"].astype(str)
        rep = rep.drop_duplicates("image_id").set_index("image_id")
    else:
        rep.index = rep.index.astype(str)

    missing = [i for i in image_ids if i not in rep.index]
    if missing:
        raise ValueError(
            "Missing {} images in representations. First few: {}".format(
                len(missing), missing[:10]
            )
        )

    sub = rep.loc[image_ids]

    E = np.stack([np.asarray(v, dtype=np.float32) for v in sub[rep_col].values])

    if E.ndim == 2:
        E = E[:, None, :]

    if E.ndim != 3:
        raise ValueError("Expected shape N,T,D or N,D. Got {}".format(E.shape))

    return E, np.array(image_ids)


def cosine_distance_matrix_mean_tta(E1, E2):
    """Cross-cohort cosine distance matrix, averaged over TTA slices.

    Notebook cell 25. Rows are E1's images, columns are E2's images.
    """
    if E1.ndim != 3 or E2.ndim != 3:
        raise ValueError("E1 and E2 must be N,T,D tensors.")

    if E1.shape[1:] != E2.shape[1:]:
        raise ValueError(
            "TTA / embedding dimension mismatch: {} vs {}".format(E1.shape, E2.shape)
        )

    D_sum = None

    for t in range(E1.shape[1]):
        D_t = 1.0 - cosine_similarity(E1[:, t, :], E2[:, t, :])
        D_sum = D_t if D_sum is None else D_sum + D_t

    return D_sum / E1.shape[1]
