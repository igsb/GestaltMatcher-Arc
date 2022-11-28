import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def eval(gallery_df, gallery_set_representations, test_set_representations, test_synd_ids):
    # have to reshape the array manually due to different size repr.vec. -> [model, img, [1,dim]]
    test_set_representations = [
        np.array([test_set_representations[j][i] for j in range(len(test_set_representations))]) for i in
        range(len(test_set_representations[0]))]

    gallery_set_representations = [
        np.array([gallery_set_representations[j][i] for j in range(len(gallery_set_representations))]) for i in
        range(len(gallery_set_representations[0]))]

    # Per img, per model sorted min distance from test to gallery(index)
    dists = np.stack([pairwise_distances(test_set_representations[i], gallery_set_representations[i], 'cosine')
                      for i in range(len(test_set_representations))], axis=1)

    # Condense the model-axis to end up with 1 vote per image, rather than 1 vote per model per image
    # Note: linearly weighted vote-based system has complication
    # -> we can't increment indices as syndrome id is no longer unique ...
    # .. maybe use the gallery image id as index instead
    ranked_dists = np.argsort(np.mean(dists, axis=1), axis=1)  # average the distances over all models (BAD IDEA?)
    ranked_synds = gallery_df.values[ranked_dists][:, :, -1]

    # This removes all duplicate occurrences except for the first one.. for each test image
    guessed_all = np.array([ranked_synds[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                            range(len(ranked_synds))])  # Expected shape: [num_images_test, num_images_gallery]

    # Top-n performance
    corr = np.zeros(4)  # 4 because [1,5,10,30]
    acc_per = []
    for i, n in enumerate([1, 5, 10, 30]):
        for idx in range(len(test_synd_ids)):
            # guessed_all[np.sort(np.unique(guessed_all, return_index=True)[1])]
            top_n_guessed = guessed_all[idx, 0:n]
            if test_synd_ids[idx] in top_n_guessed:
                corr[i] += 1

        # Bit cluttered.., but this calculates the top-n per syndrome accuracy
        acc_per.append(sum([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]]) / len(
            np.where(test_synd_ids == tl)[0]) for tl in list(set(test_synd_ids))]) / len(
            list(set(test_synd_ids))))

    return acc_per


# Get syndrome id from index id
with open('lookup_table_gmdb.txt', 'r') as f:
# with open('lookup_table_gmdb_v1.0.3.txt', 'r') as f:
    line = f.readlines()[1]
    synd_lookup_table = np.array(json.loads(line))

# Get target labels
data_path = os.path.join('..', 'data', 'GestaltMatcherDB', 'v1.0.3', 'gmdb_metadata')
test_df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_test_images_v1.0.3.csv'))
frequent_test_image_ids = test_df.image_id.values

# Get frequent gallery info
representation_df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.0.3.csv'))

# Get all predictions
representation_df = pd.read_csv("all_encodings.csv", delimiter=";")
representation_df = representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()

representation_df.representations = representation_df.representations.apply(lambda x: [json.loads(i) for i in x])
representation_df.class_conf = representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])
representation_df.img_name = representation_df.img_name.apply(lambda x: int(x.split('_')[0]))

# GestaltMatcher test: Frequent, gallery: Frequent
gallery_df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.0.3.csv'))
gallery_df['synd_id'] = np.array([np.where(synd_lookup_table == sid)[0][0] for sid in gallery_df.label])
# gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

test_df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_test_images_v1.0.3.csv'))
# ids in look up table ..:
test_synd_ids = np.array([np.where(synd_lookup_table == sid)[0][0] for sid in test_df.label])
# test_synd_ids = np.array([sid for sid in test_df.label])

# Get the representations of the relevant sets
gallery_set_representations = representation_df.representations.values[
    np.nonzero(gallery_df.image_id.values[:, None] == representation_df.img_name.values)[1]]
test_set_representations = representation_df.representations.values[
    np.nonzero(test_df.image_id.values[:, None] == representation_df.img_name.values)[1]]

## Tests ##
acc_per = eval(gallery_df, gallery_set_representations, test_set_representations, test_synd_ids)
print('===========================================================')
print('---------   test: Frequent, gallery: Frequent    ----------')
print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
print('|{}|{}    |{}   |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-frequent",
                                                                          len(gallery_set_representations),
                                                                          len(test_set_representations),
                                                                          (acc_per[0]) * 100,
                                                                          (acc_per[1]) * 100,
                                                                          (acc_per[2]) * 100,
                                                                          (acc_per[3]) * 100))

# GestaltMatcher test: Rare, gallery: Rare
# Note: the syndrome ids are not in the lookup table, as they weren't part of the training set
gallery_df = pd.read_csv(os.path.join(data_path, 'gmdb_rare_gallery_images_v1.0.3.csv'))
gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

test_df = pd.read_csv(os.path.join(data_path, 'gmdb_rare_test_images_v1.0.3.csv'))

acc_per_list = []
num_splits = max(gallery_df.split) + 1
for test_split in range(num_splits):
    # ids in look up table ..:
    test_synd_ids = np.array([sid for sid in test_df[test_df.split == test_split].label])
    gallery_df_split = gallery_df[gallery_df.split == test_split]

    # Get the representations of the relevant sets
    gallery_set_representations = representation_df.representations.values[
        np.nonzero(
            gallery_df[gallery_df.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
            1]]
    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df[test_df.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
            1]]

    acc_per_list.append(eval(gallery_df_split, gallery_set_representations, test_set_representations, test_synd_ids))

acc_per_list = np.array(acc_per_list)
print('---------       test: Rare, gallery: Rare        ----------')
print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
print('|{}|{}   |{} |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-rare    ",
                                                                       len(gallery_df) / num_splits,
                                                                       len(test_df) / num_splits,
                                                                       np.mean(acc_per_list[:, 0]) * 100,
                                                                       np.mean(acc_per_list[:, 1]) * 100,
                                                                       np.mean(acc_per_list[:, 2]) * 100,
                                                                       np.mean(acc_per_list[:, 3]) * 100))


# GestaltMatcher test: Frequent, gallery: Frequent+Rare
# Note: the syndrome ids are not in the lookup table, as they weren't part of the training set
gallery_df1 = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.0.3.csv'))
gallery_df2 = pd.read_csv(os.path.join(data_path, 'gmdb_rare_gallery_images_v1.0.3.csv'))
gallery_df = gallery_df1.append(gallery_df2)
gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

test_df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_test_images_v1.0.3.csv'))

acc_per_list = []
num_splits = max(gallery_df.dropna().split) + 1
for test_split in range(int(num_splits)):
    # ids in look up table ..:
    test_synd_ids = np.array([sid for sid in test_df.label])
    gallery_df_split = gallery_df.fillna(test_split)
    gallery_df_split = gallery_df_split[gallery_df_split.split == test_split].reset_index()

    # Get the representations of the relevant sets
    gallery_set_representations = representation_df.representations.values[
        np.nonzero(
            gallery_df_split[gallery_df_split.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
            1]]
    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df.image_id.values[:, None] == representation_df.img_name.values)[
            1]]

    acc_per_list.append(eval(gallery_df_split, gallery_set_representations, test_set_representations, test_synd_ids))

acc_per_list = np.array(acc_per_list)

print('--------- test: Frequent, gallery: Frequent+Rare ----------')
print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
print('|{}|{}  |{}   |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-frequent",
                                                                       (len(gallery_df1) + len(gallery_df2) / num_splits),
                                                                       len(test_df),
                                                                       np.mean(acc_per_list[:, 0]) * 100,
                                                                       np.mean(acc_per_list[:, 1]) * 100,
                                                                       np.mean(acc_per_list[:, 2]) * 100,
                                                                       np.mean(acc_per_list[:, 3]) * 100))


# GestaltMatcher test: Rare, gallery: Frequent+Rare
gallery_df1 = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.0.3.csv'))
gallery_df2 = pd.read_csv(os.path.join(data_path, 'gmdb_rare_gallery_images_v1.0.3.csv'))
gallery_df = gallery_df1.append(gallery_df2)
gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

test_df = pd.read_csv(os.path.join(data_path, 'gmdb_rare_test_images_v1.0.3.csv'))

acc_per_list = []
num_splits = max(gallery_df.dropna().split) + 1
for test_split in range(int(num_splits)):
    # ids in look up table ..:
    test_synd_ids = np.array([sid for sid in test_df[test_df.split == test_split].label])
    gallery_df_split = gallery_df.fillna(test_split)
    gallery_df_split = gallery_df_split[gallery_df_split.split == test_split].reset_index()

    # Get the representations of the relevant sets
    gallery_set_representations = representation_df.representations.values[
        np.nonzero(
            gallery_df_split[gallery_df_split.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
            1]]
    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df[test_df.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
            1]]

    acc_per_list.append(eval(gallery_df_split, gallery_set_representations, test_set_representations, test_synd_ids))

acc_per_list = np.array(acc_per_list)

print('---------   test: Rare, gallery: Frequent+Rare   ----------')
print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
print('|{}|{}  |{} |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-rare    ",
                                                                       (len(gallery_df1) + len(gallery_df2) / num_splits),
                                                                       len(test_df) / num_splits,
                                                                       np.mean(acc_per_list[:, 0]) * 100,
                                                                       np.mean(acc_per_list[:, 1]) * 100,
                                                                       np.mean(acc_per_list[:, 2]) * 100,
                                                                       np.mean(acc_per_list[:, 3]) * 100))
print('===========================================================')
