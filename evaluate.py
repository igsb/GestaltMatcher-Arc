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

    parser.add_argument('--top_n', default=5, dest='top_n',
                        help='Top-N results to show per image/encoding')

    parser.add_argument('--case_list', default=[], dest='case_list', nargs='*', #nargs='+',
                        help='List of case encodings to use. Default: []')
    parser.add_argument('--gallery_list', default=[], dest='gallery_list', nargs='*', #nargs='+',
                        help='List of gallery encodings to use. Default: []')
    return parser.parse_args()
args = parse_args()


# TODO: This is the direction we're probably heading into w.r.t. the script
# def evaluate(gallery_df, case_df, metadata_dir=''):
#     synds = pd.read_csv(os.path.join(metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
#                         delimiter='\t',
#                         usecols=['syndrome_id', 'syndrome_name'])
#
#     # TODO: Can be removed as we don't use the lookup table, UNLESS we want to use the class_confidence. If we do want
#     #  to use the classification confidence, note that the non-finetuned models don't provide you one.
#     # Get syndrome id from index id
#     with open(args.lut, 'r') as f:
#         line = f.readlines()[1]
#         synd_lookup_table = np.array(json.loads(line))
#
#     # TODO: Gallery encodings (gallery_df) need to be linked to their syndrome_ids (listed below as gallery_df.label,
#     #  but likely you want to get them either before `evaluate.py` or from a different location like somewhere in the
#     #  metadata_dir).
#     gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])
#     gallery_df.image_id = gallery_df.image_id.astype(str)
#
#     # Get representations of just the gallery set
#     gallery_set_representations = gallery_df.representations.values
#     gallery_set_representations = np.stack(gallery_set_representations)
#
#     # Get representations of just the gallery set
#     case_representations = case_df.representations.values
#     case_representations = np.stack(case_representations)
#
#     # Actually get distances
#     def eval(gallery_df, gallery_set_representations, test_set_representations):
#         def reshape_representations(representations):
#             representations = [
#                 np.array([representations[j][i] for j in range(len(representations))]) for i in
#                 range(len(representations[0]))]
#             return representations
#
#         # have to reshape the array manually due to different size repr.vec. -> [model/tta, img, [1,dim]]
#         test_set_representations = reshape_representations(test_set_representations)
#         gallery_set_representations = reshape_representations(gallery_set_representations)
#
#         # Per img, per model/tta sorted min distance from test to gallery(index)
#         dists = np.stack(
#             [pairwise_distances(test_set_representations[model_tta], gallery_set_representations[model_tta], 'cosine')
#              for model_tta in range(len(test_set_representations))], axis=1)
#         mean_dists = np.mean(dists, axis=1)
#
#         # Condense the model-axis to end up with 1 vote per image, rather than 1 vote per model per image
#         ranks_dists = np.argsort(mean_dists, axis=1)  # average the distances over all models
#         ranked_mean_dists = np.take_along_axis(mean_dists, ranks_dists, axis=1)
#         ranked_synd_ids = gallery_df.values[ranks_dists][:, :, -1]
#         ranked_img_ids = gallery_df.values[ranks_dists][:, :, 0]
#         ranked_subject_ids = gallery_df.values[ranks_dists][:, :, 1]
#
#         return ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids
#
#     # return ranked_synd_ids, ranked_mean_dists, ranked_img_ids, ranked_subject_ids
#     return eval(gallery_df, gallery_set_representations, case_representations)


# Used to belong to evaluate.py - keeping a backup here in case we need it ...
def evaluate(gallery='all', metadata_dir=''):
    synds = pd.read_csv(os.path.join(metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
                        delimiter='\t',
                        usecols=['syndrome_id', 'syndrome_name'])

    # Get syndrome id from index id
    with open(args.lut, 'r') as f:
        line = f.readlines()[1]
        synd_lookup_table = np.array(json.loads(line))

    def prep_csv(df):
        df = pd.read_csv(df, delimiter=';')
        df = df.groupby('img_name').agg(lambda x: list(x)).reset_index()
        df.representations = df.representations.apply(lambda x: np.array([json.loads(i) for i in x]))
        #df.class_conf = df.class_conf.apply(lambda x: [json.loads(i) for i in x])
        df.img_name = df.img_name.apply(lambda x: x.split('_')[0])
        return df

    # Get all predictions
    all_df = prep_csv(os.path.join(args.gallery_dir, "all_encodings.csv"))  # maybe rename in the future ...

    # Get case predictions
    case_df = prep_csv(os.path.join(args.case_dir, "case_encodings.csv"))

    # Get gallery set info
    gallery_df1 = pd.read_csv(os.path.join(metadata_dir, 'gmdb_frequent_gallery_images_v1.0.3.csv'))
    gallery_df2 = pd.read_csv(os.path.join(metadata_dir, 'gmdb_rare_gallery_images_v1.0.3.csv'))\
        .drop("split", axis=1).drop_duplicates()    # remove 'split'-column and then remove duplicates
    if gallery in ['all', 'unified', 'freq+rare', 'rare+freq']:
        gallery_df = gallery_df1.append(gallery_df2)
    elif gallery == 'freq':
        gallery_df = gallery_df1
    elif gallery == 'rare':
        gallery_df = gallery_df2
    else:
        print(f'Unrecognised gallery options ("{gallery}" selected).')
        exit()
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])
    gallery_df.image_id = gallery_df.image_id.astype(str)
    del gallery_df1, gallery_df2

    # Get representations of just the gallery set
    gallery_set_representations = all_df.representations.values[
        np.nonzero(gallery_df.image_id.values[:, None] == all_df.img_name.values)[1]
    ]
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
            print(list(np.array([synds.iloc[id].syndrome_name for id in aa])[:5]))

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
    if os.path.splitext(encoding_files[0])[-1] == '.pkl':
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


def print_format_output(results):
    synd_ids = results[0][0]
    dists = results[1][0]
    img_ids = results[2][0]
    subject_ids = results[3][0]
    print(f"Synd ids: {list(synd_ids)}")
    print(f"Distances: {[round(i, 3) for i in dists]}")
    print(f"Image ids: {[int(i) for i in img_ids]}")
    print(f"Subject ids: {list(subject_ids)}")


def main():

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Location of the GMDB meta data
    if args.metadata_dir == '':
        args.metadata_dir = os.path.join('..', 'data', 'GestaltMatcherDB', 'v1.0.3', 'gmdb_metadata')

    # Load and combine all encodings
    # args.separate_files_gallery
    gallery_df = get_encodings_set(args.gallery_dir, args.gallery_list)

    # args.separate_files_cases
    case_df = get_encodings_set(args.case_dir, args.case_list)

    ## Evaluate
    # Get all synd_ids, dists, img_ids, subject_ids per image in gallery
    n = int(args.top_n)

    args.gallery_preset = 'rare+freq'
    # all_ranks = evaluate(gallery_df=gallery_df, case_df=case_df, metadata_dir=args.metadata_dir)
    all_ranks = evaluate("all", metadata_dir=args.metadata_dir)
    all_ranks = np.array(all_ranks)

    start = time.time()
    # TEST PRINT DISORDER NAMES
    stuff = all_ranks[0,0,:n]
    synds = pd.read_csv(os.path.join(args.metadata_dir, 'gmdb_syndromes_v1.0.3.tsv'),
                        delimiter='\t',
                        usecols=['syndrome_id', 'syndrome_name'])
    print(f"Top-{n} disorders:")
    for aa in stuff:
        print(f"{synds.iloc[aa].syndrome_name}")

    print(f"\nTop-{n} results on image level:")
    print_format_output(all_ranks[:,:,:n])

    # Get all synd_ids, dists, img_ids, subject_ids per syndrome in gallery
    first_synd_ranks = get_first_synds(*all_ranks)
    first_synd_ranks = np.array(first_synd_ranks)
    print(f"\nTop-{n} results on syndrome level:")
    print_format_output(first_synd_ranks[:, :, :n])

    # Get all synd_ids, dists, img_ids, subject_ids per subject in gallery
    first_subject_ranks = get_first_subject(*all_ranks)
    first_subject_ranks = np.array(first_subject_ranks)
    print(f"\nTop-{n} results on subject level:")
    print_format_output(first_subject_ranks[:, :, :n])



if __name__ == '__main__':
    main()
