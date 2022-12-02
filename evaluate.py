## predict_evaluate_cases.py
# Run the chosen models on every image in the desired directory
# and save the encodings to the file: "case_encodings.csv"
# also run the evaluation-script between the "encodings.csv" (of the gallery set)
# and the new case encodings

import argparse
import json
import os
import random
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances


def parse_args():
    parser = argparse.ArgumentParser(description='Predict DeepGestalt')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    parser.add_argument('--lut', default='lookup_table_gmdb.txt', dest='lut',
                        help='Path to the lookup table.')

    parser.add_argument('--case_dir', default='data', dest='case_dir',
                        help='Path to the directory containing the case encodings.')
    parser.add_argument('--metadata_dir', default='', dest='metadata_dir',
                        help='Path to the directory containing the gallery set metadata.')
    parser.add_argument('--gallery_dir', default='', dest='gallery_dir',
                        help='Path to the directory containing the gallery encodings.')

    parser.add_argument('--case_list', default=[], dest='case_list', nargs='*', #nargs='+',
                        help='List of case encodings to use. Default: []')
    parser.add_argument('--gallery_list', default=[], dest='gallery_list', nargs='*', #nargs='+',
                        help='List of gallery encodings to use. Default: []')

    # parser.add_argument('--separate_files_gallery', action='store_true', default=False,
    #                     help='When set expects a list of csv-files for the gallery using \'--gallery_list\'.')
    # parser.add_argument('--separate_files_cases', action='store_true', default=False,
    #                     help='When set expects a list of csv-files for the cases using \'--case_list\'.')
    return parser.parse_args()
args = parse_args()


def evaluate(gallery_df, case_df, metadata_dir=''):
    synds = pd.read_csv(os.path.join(metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
                        delimiter='\t',
                        usecols=['syndrome_id', 'syndrome_name'])

    # TODO: Can be removed as we don't use the lookup table, UNLESS we want to use the class_confidence. If we do want
    #  to use the classification confidence, note that the non-finetuned models don't provide you one.
    # Get syndrome id from index id
    with open(args.lut, 'r') as f:
        line = f.readlines()[1]
        synd_lookup_table = np.array(json.loads(line))

    # TODO: Gallery encodings (gallery_df) need to be linked to their syndrome_ids (listed below as gallery_df.label,
    #  but likely you want to get them either before `evaluate.py` or from a different location like somewhere in the
    #  metadata_dir).
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])
    gallery_df.image_id = gallery_df.image_id.astype(str)

    # Get representations of just the gallery set
    gallery_set_representations = gallery_df.representations.values
    gallery_set_representations = np.stack(gallery_set_representations)

    # Get representations of just the gallery set
    case_representations = case_df.representations.values
    case_representations = np.stack(case_representations)

    # Actually get distances
    def eval(gallery_df, gallery_set_representations, test_set_representations):
        def reshape_representations(representations):
            representations = [
                np.array([representations[j][i] for j in range(len(representations))]) for i in
                range(len(representations[0]))]
            return representations

        # have to reshape the array manually due to different size repr.vec. -> [model/tta, img, [1,dim]]
        test_set_representations = reshape_representations(test_set_representations)
        gallery_set_representations = reshape_representations(gallery_set_representations)

        # Per img, per model/tta sorted min distance from test to gallery(index)
        dists = np.stack(
            [pairwise_distances(test_set_representations[model_tta], gallery_set_representations[model_tta], 'cosine')
             for model_tta in range(len(test_set_representations))], axis=1)
        mean_dists = np.mean(dists, axis=1)

        # Condense the model-axis to end up with 1 vote per image, rather than 1 vote per model per image
        ranks_dists = np.argsort(mean_dists, axis=1)  # average the distances over all models
        ranked_mean_dists = np.take_along_axis(mean_dists, ranks_dists, axis=1)
        ranked_synd_ids = gallery_df.values[ranks_dists][:, :, -1]
        ranked_img_ids = gallery_df.values[ranks_dists][:, :, 0]
        ranked_subject_ids = gallery_df.values[ranks_dists][:, :, 1]

        return ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids

    # return ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids
    return eval(gallery_df, gallery_set_representations, case_representations)


def get_first_synds(ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids, verbose=False):
    # This removes all duplicate occurrences except for the first one.. for each test image
    synds_all = np.array([ranked_synd_ids[i][np.sort(np.unique(ranked_synd_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_synd_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    dists_all = np.array([ranked_mean_dists[i][np.sort(np.unique(ranked_synd_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_synd_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    imgs_all = np.array([ranked_img_ids[i][np.sort(np.unique(ranked_synd_ids[i], return_index=True)[1])] for i in
                         range(len(ranked_synd_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    subjects_all = np.array(
        [ranked_subject_ids[i][np.sort(np.unique(ranked_synd_ids[i], return_index=True)[1])] for i in
         range(len(ranked_synd_ids))])  # Expected shape: [num_images_test, num_images_gallery]

    if verbose:
        synds = pd.read_csv(os.path.join(args.metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
                            delimiter='\t',
                            usecols=['syndrome_id', 'syndrome_name'])
        for aa in ranked_img_ids:
            print(list(aa))
            print(list(np.array([synds.iloc[id].syndrome_name for id in aa])[:10]))

    return synds_all, dists_all, imgs_all, subjects_all


def get_first_subject(ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids, verbose=False):
    # This removes all duplicate occurrences except for the first one.. for each test image
    synds_all = np.array([ranked_synd_ids[i][np.sort(np.unique(ranked_subject_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_subject_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    dists_all = np.array([ranked_mean_dists[i][np.sort(np.unique(ranked_subject_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_subject_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    imgs_all = np.array([ranked_img_ids[i][np.sort(np.unique(ranked_subject_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_subject_ids))])  # Expected shape: [num_images_test, num_images_gallery]
    subjects_all = np.array([ranked_subject_ids[i][np.sort(np.unique(ranked_subject_ids[i], return_index=True)[1])] for i in
                          range(len(ranked_subject_ids))])  # Expected shape: [num_images_test, num_images_gallery]

    if verbose:
        synds = pd.read_csv(os.path.join(args.metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
                            delimiter='\t',
                            usecols=['syndrome_id', 'syndrome_name'])
        for aa in ranked_img_ids:
            print(list(aa))
            print(list(np.array([synds.iloc[id].syndrome_name for id in aa])[:10]))

    return synds_all, dists_all, imgs_all, subjects_all


def get_encodings_set(encoding_dir, encoding_files):
    # Helper function to correct DataFrame
    def prep_csv(df):
        df = df.groupby('img_name').agg(lambda x: list(x)).reset_index()
        df.representations = df.representations.apply(lambda x: np.array([json.loads(i) for i in x]))
        #df.class_conf = df.class_conf.apply(lambda x: [json.loads(i) for i in x])
        df.img_name = df.img_name.apply(lambda x: x.split('_')[0])
        return df

    # First check if we're dealing with *.pkl-files or *.csv-files
    is_pickle = False
    if encoding_files[0].split('.')[-1] == 'pkl':
        is_pickle = True

    # Next check if we use one large file or several small ones
    is_separate = False
    if len(encoding_files) > 1:
        is_separate = True

    # Finally, check if it's a directory instead of a file
    if os.path.isdir(os.path.join(encoding_dir, encoding_files[0])):
        is_dir = True
        is_separate = True

    df_main = None
    # single file
    if not is_separate:
        if is_pickle:
            df_main = pd.read_pickle(os.path.join(encoding_dir, encoding_files[0]))
        else:  # is csv
            df_main = prep_csv(pd.read_csv(os.path.join(encoding_dir, encoding_files[0]), delimiter=';'))
        return df_main

    # else: separate files or dir
    if is_dir:
        encoding_files = glob(f"{os.path.join(os.path.join(encoding_dir, encoding_files[0]))}/*")
    for file in encoding_files:
        if is_pickle:
            df_part = pd.read_pickle(os.path.join(encoding_dir, file))
        else:  # is csv
            df_part = prep_csv(pd.read_csv(os.path.join(encoding_dir, file), delimiter=';'))
        df_main = pd.concat([df_main, df_part])
    return prep_csv(df_main)


def main():

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Location of the GMDB meta data
    if args.metadata_dir == '':
        args.metadata_dir = os.path.join('..', '..', '..', 'data', 'GestaltMatcherDB', 'v1.0.3', 'gmdb_metadata')

    # Load and combine all encodings
    # args.separate_files_gallery
    gallery_df = get_encodings_set(args.gallery_dir, args.gallery_list)

    # args.separate_files_cases
    case_df = get_encodings_set(args.case_dir, args.case_list)

    ## Evaluate
    # Get all synd_ids, dists, img_ids, subject_ids per image in gallery
    args.gallery_preset = 'rare+freq'
    # all_ranks = evaluate(gallery=args.gallery_preset, metadata_dir=args.metadata_dir)
    all_ranks = evaluate(gallery_df=gallery_df, case_df=case_df, metadata_dir=args.metadata_dir)
    all_ranks = np.array(all_ranks)
    print(f"Top-5 results img_id (synd_id, dist, img_id, subject_id):\n{all_ranks[:,0,:5]}")

    # Get all synd_ids, dists, img_ids, subject_ids per syndrome in gallery
    first_synd_ranks = get_first_synds(*all_ranks)
    first_synd_ranks = np.array(first_synd_ranks)
    print(f"Top-5 synd_id results (synd_id, dist, img_id, subject_id):\n{first_synd_ranks[:,0,:5]}")

    # Get all synd_ids, dists, img_ids, subject_ids per subject in gallery
    first_subject_ranks = get_first_subject(*all_ranks)
    first_subject_ranks = np.array(first_subject_ranks)
    print(f"Top-5 subject_id results (synd_id, dist, img_id, subject_id):\n{first_subject_ranks[:,0,:5]}")

if __name__ == '__main__':
    main()


## TODO - work this into the eval ...
# dir = "../all"
# csvs = glob(f"{dir}/*")
#
# total_df = None
# tic = time.time()
# for csv in csvs:
#   df = pd.read_csv(csv, delimiter=';')
#   total_df = pd.concat([total_df, df])
#
# time.time()-tic
#
# for csv in csvs:
#   df = pd.read_csv(csv, delimiter=';')
#   csv_name = (csv.split('\\')[-1]).split('.')[0]
#   df.to_pickle(f"{dir2}/{csv_name}.pkl")
#
#
# total_df = None
# tic = time.time()
# for pkl in pkls:
#   df = pd.read_pickle(pkl)
#   total_df = pd.concat([total_df, df])
#
# time.time()-tic