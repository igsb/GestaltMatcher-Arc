import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate (Multi-Image) GestaltMatcher')

    parser.add_argument('--split', dest='eval_split', default='all',
                        help='Test and gallery splits to use. Options: all, ff, rr, fa, ra. (default=all)')
    parser.add_argument('--version', dest='version', default='v1.1.0',
                        help='GMDB version to evaluate. (default=v1.1.0)')
    parser.add_argument('--multi_only', dest='multi_only', action='store_true',
                        help='Flag to set if you only want to evaluate on multiple images. (default=false)')

    parser.add_argument('--encodings_path', dest='encodings_path', default='./encodings/all_encodings.csv',
                        help='Path to the file containing GM encodings of all images. Supported types: .csv and .pkl (default=./encodings/all_encodings.csv)')
    parser.add_argument('--metadata_path', dest='data_path', default='../data/GestaltMatcherDB/v1.1.0/gmdb_metadata',
                        help='Path to the directory containing metadata-files. (default=../data/GestaltMatcherDB/v1.1.0/gmdb_metadata)')

    return parser.parse_args()


def print_util(dicto, tops=[1,5,10,30]):
    keys = list(dicto.keys())

    for key in keys:
        value = dicto[key][tops]
        key_string = key.replace(', ', '\t')[1:-1]
        value_string = np.array2string(value, separator='\t')[1:-1]
        print(f'{key_string}\t{value_string}')

def mean_accs(accs_dicts):
    accs_dict_split = {}
    for accs_dict in accs_dicts:
        for k in accs_dict.keys():
            if k not in accs_dict_split:
                accs_dict_split[k] = 0.
            accs_dict_split[k] += accs_dict[k] / len(accs_dicts)
    return accs_dict_split

def eval_all(
        gallery_df,
        gallery_set_representations,
        test_set_representations,
        test_synd_ids,
        test_set_patients=None
):
    all_results = {}

    gallery_set_representations = np.array([
        np.array([gallery_set_representations[j][i] for j in range(len(gallery_set_representations))]) for i in
        range(len(gallery_set_representations[0]))])

    # have to reshape the array manually due to different size repr.vec. -> [model, img, [1,dim]]
    test_set_representations = np.array([
        np.array([test_set_representations[j][i] for j in range(len(test_set_representations))]) for i in
        range(len(test_set_representations[0]))])

    # Reset the index of the gallery_df for continuity (in Rare we subset the gallery_df)
    gallery_df = gallery_df.reset_index(drop=True)

    def loop_code(
            gallery_df,
            gallery_set_representations,
            test_set_representations,
            test_synd_ids,
            test_set_patients,
            fuse_patient_gallery,
            fuse_patient_test,
            cluster_disorders,
            tta_approach='vanilla',
            experiment='none',
            fuse_func=np.mean,
            num_clusters=1
    ):
        if fuse_func == np.max and experiment != 'none':
            return [0]

        # Gallery labels
        gallery_synd_ids = gallery_df.synd_id.values
        num_synds = len(np.unique(gallery_synd_ids))

        if fuse_patient_gallery:
            ## Early fusion of gallery patient representations
            # (e.g., when merged image representations first, and then clustering disorders)
            # expected shape: [12, num_patients_gallery, 512d]
            patient_ids = gallery_df.subject.unique()
            patient_id_to_synd = np.array([gallery_df[gallery_df.subject == patient_id].synd_id.values[0] for patient_id in patient_ids])
            gallery_set_patient_idxs = [list(gallery_df[gallery_df.subject == patient_id].index) for patient_id in patient_ids]
            # gallery_set_patient_representations = np.array([np.mean(gallery_set_representations[:, patient_idxs], axis=1) for patient_idxs in gallery_set_patient_idxs]).swapaxes(0,1)
            gallery_set_representations = np.array([fuse_func(gallery_set_representations[:, patient_idxs], axis=1) for patient_idxs in gallery_set_patient_idxs]).swapaxes(0,1)

            gallery_synd_ids = patient_id_to_synd

        if fuse_patient_test and test_set_patients is not None:
            # Early fusion of test patient representations
            # expected shape: [12, num_patients_test, 512d]
            test_patient_ids = np.unique(test_set_patients)
            test_set_patient_idxs = [np.where(test_set_patients == patient_id)[0] for patient_id in test_patient_ids]
            test_set_representations = np.array([fuse_func(test_set_representations[:, patient_idxs], axis=1) for patient_idxs in test_set_patient_idxs]).swapaxes(0, 1)
            test_synd_ids = np.array([np.mean(test_synd_ids[patient_idxs], axis=0) for patient_idxs in test_set_patient_idxs])  # np.mean out of laziness.. should work

        if tta_approach == 'vanilla' or tta_approach == 'late_fusion' or tta_approach is None:
            # normal approach
            pass
        elif tta_approach == 'early_fusion':
            ## Early fusion: combine representations per model
            # expected shape: [num_models, num_images, 512d]; num_models = 3
            gallery_set_representations = np.stack([np.mean(gallery_set_representations[x:y], axis=0) for x,y in [[0,4], [4,8], [8,12]]])
            test_set_representations = np.stack([np.mean(test_set_representations[x:y], axis=0) for x,y in [[0,4], [4,8], [8,12]]])

        ## Get clusters
        if cluster_disorders:
            idx_per_synd = []
            gallery_cluster_representations = []
            gallery_synd_ids = []
            for synd_id in list(set(gallery_df.synd_id.values)):
                if fuse_patient_gallery:
                    idxs = np.where(patient_id_to_synd == synd_id)[0]
                else:
                    idxs = list(gallery_df[gallery_df.synd_id == synd_id].index)
                idx_per_synd.append(idxs)

                if num_clusters == 1:  # 1 centroid per disorder: just mean all
                    gallery_cluster_representations.append(np.mean(gallery_set_representations[:,idxs], axis=1))
                    gallery_synd_ids.append(synd_id)

            gallery_set_representations = np.swapaxes(np.array(gallery_cluster_representations),0,1).reshape([gallery_cluster_representations[0].shape[0], -1, 512])

        # Per img, per 'model' compute cosine distance from test to gallery
        # Note: also works if we've already fused the models' representations, as long as it is a dimension at axis 0
        dists = np.stack([pairwise_distances(test_set_representations[i], gallery_set_representations[i], 'cosine')
                          for i in range(len(test_set_representations))], axis=0)

        # Condense the model-axis to end up with 1 distance per image, rather than 1 distance vote per model per image
        ranked_dists = np.argsort(np.mean(dists, axis=0), axis=1)
        ranked_synds = np.array([[gallery_synd_ids[idx] for idx in ranked_dists[i]] for i in range(len(ranked_dists))])

        avg_dists = np.sort(np.mean(dists, axis=0), axis=1)

        # Experiment: Rank averaging
        if experiment == 'exp_rank_averaging':
            if test_set_patients is not None and not fuse_patient_test:
                ranked_synd_ids_unique = np.stack([np.argsort([np.unique(a, return_index=True)[1]]) for a in ranked_synds])
                correct_ranks_per_image = np.array([np.where(ranked_synd_ids_unique[i] == np.where(np.unique(gallery_synd_ids) == test_synd_ids[i])[0][0])[1][0] for i in range(len(test_synd_ids))])
                test_patient_ids = np.unique(test_set_patients)
                test_set_patient_idxs = [np.where(test_set_patients == patient_id)[0] for patient_id in test_patient_ids]
                rank_average_per_patient = []
                for patient_idxs in test_set_patient_idxs:
                    # Experiment: get rank-based performance for every test image
                    rank_average_per_patient.append(np.mean(correct_ranks_per_image[patient_idxs]))
                rank_average_per_patient = np.array(rank_average_per_patient)
                rank_average_top_n = np.array([np.sum(rank_average_per_patient<n)/len(rank_average_per_patient) for n in range(num_synds)])
                return rank_average_top_n
            else:
                return np.zeros(31)

        # Keep only the best match per disorder (i.e., top-1,2,3 cannot all be the same synd_id)
        if (cluster_disorders == False) or (num_clusters != 1):
            # This removes all duplicate occurrences except for the closest one, for each test image
            ranked_dists = np.array([ranked_dists[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                                    range(len(ranked_synds))])
            avg_dists = np.array([avg_dists[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                                    range(len(ranked_synds))])
            ranked_synds = np.array([ranked_synds[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                                    range(len(ranked_synds))])  # Expected shape: [num_images_test, num_images_gallery]

        # Experiment: Late fusion test patients
        if experiment == 'exp_late_fuse_test':
            if test_set_patients is not None and not fuse_patient_test:
                # Get lowest cosine distance per syndrome, per test patient
                best_dist_by_synd = np.array([avg_dists[i][np.argsort(ranked_synds[i])] for i in range(len(ranked_synds))])

                # Get indices of test images for each test patient
                test_patient_ids = np.unique(test_set_patients)
                test_set_patient_idxs = [np.where(test_set_patients == patient_id)[0] for patient_id in test_patient_ids]

                # Compute the mean cosine distance of the best matches per syndrome, per test patient
                late_fusion_mean_dist_per_patient = np.array([np.mean(best_dist_by_synd[idxs], axis=0) for idxs in test_set_patient_idxs])

                # Compute a new syndrome ranking per test patient and it's top-n performance
                ranked_synds_late_fusion = np.argsort(late_fusion_mean_dist_per_patient)
                correct_ranks_per_test_patient = np.array([np.where(ranked_synds_late_fusion[i] == np.where(np.unique(gallery_synd_ids) == test_synd_ids[test_set_patient_idxs[i][0]])[0][0])[0][0] for i in range(len(ranked_synds_late_fusion))])
                late_fusion_top_n = np.array([np.sum(correct_ranks_per_test_patient < n) / len(correct_ranks_per_test_patient) for n in range(num_synds)])
                return late_fusion_top_n
            else:
                return np.zeros(31)

        corr = np.zeros(len(list(set(ranked_synds[0]))) + 1)
        acc_per = []
        for i, n in enumerate(range(len(list(set(ranked_synds[0]))))):
            for idx in range(len(test_synd_ids)):
                # guessed_all[np.sort(np.unique(guessed_all, return_index=True)[1])]
                top_n_guessed = ranked_synds[idx, 0:n]
                if test_synd_ids[idx] in top_n_guessed:
                    corr[i] += 1

            # Bit cluttered, but this calculates the top-n per syndrome accuracy
            acc_per.append(sum([sum(tl in g[0:n] for g in ranked_synds[np.where(test_synd_ids == tl)[0]]) / len(
                np.where(test_synd_ids == tl)[0]) for tl in list(set(test_synd_ids))]) / len(
                list(set(test_synd_ids))))
        acc_per = np.array(acc_per)
        return acc_per


    for tta_approach in ['vanilla']:#, 'early_fusion']:
        for fuse_gallery in [True, False]:
            for fuse_test in [True, False]:
                for cluster_K in [0, 1]:
                    for fuse_func in [np.mean]: #[np.mean, np.max, np.median, np.min]
                        for experiment in ['none', 'exp_rank_averaging', 'exp_late_fuse_test']:
                            tag = f'{tta_approach, fuse_gallery, fuse_test, True if cluster_K else False, cluster_K, experiment, fuse_func.__name__}'
                            # print(tag)
                            res = loop_code(
                                gallery_df, gallery_set_representations, test_set_representations, test_synd_ids, test_set_patients,
                                fuse_patient_gallery=fuse_gallery,
                                fuse_patient_test=fuse_test,
                                cluster_disorders=True if cluster_K else False,
                                tta_approach=tta_approach,
                                experiment=experiment,
                                fuse_func=fuse_func,
                                num_clusters=cluster_K
                            )
                            if sum(res) > 0:
                                all_results[tag] = res
    return all_results

args = parse_args()

print(f"Evaluating split: {args.eval_split}")

multiple_test_image_patients_only = args.multi_only
print(f"{'Evaluation only photos of patients with multiple images' if multiple_test_image_patients_only else 'Evaluation all patient photos'}")

version = args.version
print(f"Using GMDB version {version}")

# Set overall datapath for the metadata files
# data_path = os.path.join('..', 'data', 'GestaltMatcherDB', version, 'gmdb_metadata')
data_path = args.data_path

# Get all predictions
file_ext = args.encodings_path.split('.')[-1]
if file_ext == 'csv':
    representation_df = pd.read_csv(args.encodings_path, delimiter=";")
    representation_df = representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()
    representation_df.representations = representation_df.representations.apply(lambda x: [json.loads(i) for i in x])
    representation_df.class_conf = representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])
    representation_df.img_name = representation_df.img_name.apply(lambda x: int(x.split('_')[0]))
elif file_ext == 'pkl':
    representation_df = np.load(args.encodings_path, allow_pickle=True)
    representation_df.img_name = representation_df.img_name.apply(lambda x: int(x.split('_')[0]))
    representation_df = representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()
elif file_ext == 'p':
    # this pickle contains a dict of shape {img_id:[12 x representations]}
    pickle_dict = np.load(args.encodings_path, allow_pickle=True)
    data = []
    for img_id, representations in pickle_dict.items():
        data.append({'img_name': img_id, 'representations': representations})
    representation_df = pd.DataFrame(data)
else:
    raise ValueError(f'Unsupported encodings-file type; only csv- and pkl-files are supported. (got {args.encodings_path.split(".")[-1]})')


## GestaltMatcher test: Frequent, gallery: Frequent
accs_dict_ff = {}
if args.eval_split == 'all' or args.eval_split == 'ff':
    print("FREQ->FREQ")
    gallery_df = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_gallery_images_{version}.csv'))
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])
    test_df = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_test_images_{version}.csv'))

    # Get the representations of the relevant sets
    gallery_set_representations = representation_df.representations.values[
        np.nonzero(gallery_df.image_id.values[:, None] == representation_df.img_name.values)[1]]

    # keep only images of test patients with multiple images
    if multiple_test_image_patients_only:
        unq, counts = np.unique(test_df.subject.values, return_counts=True)
        pat_ids_multi = unq[counts > 1]
        test_df = test_df[test_df.subject.isin(pat_ids_multi)]

    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df.image_id.values[:, None] == representation_df.img_name.values)[1]]
    test_set_patients = test_df.subject.values

    test_synd_ids = np.array([sid for sid in test_df.label])

    ## Test(s) ##
    accs_dict_ff = eval_all(
        gallery_df,
        gallery_set_representations,
        test_set_representations,
        test_synd_ids,
        test_set_patients
    )
    print_util(accs_dict_ff)

## GestaltMatcher test: Rare, gallery: Rare
accs_dict_rr = {}
if args.eval_split == 'all' or args.eval_split == 'rr':
    print("RARE->RARE")
    gallery_df = pd.read_csv(os.path.join(data_path, f'gmdb_rare_gallery_images_{version}.csv'))
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

    test_df = pd.read_csv(os.path.join(data_path, f'gmdb_rare_test_images_{version}.csv'))

    accs_dicts = []
    num_splits = max(gallery_df.split) + 1
    for test_split in range(num_splits):
        # ids in look up table ..:
        gallery_df_split = gallery_df[gallery_df.split == test_split]
        test_df_split = test_df[test_df.split == test_split]

        # Get the representations of the relevant sets
        gallery_set_representations = representation_df.representations.values[
            np.nonzero(
                gallery_df[gallery_df.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
                1]]

        # keep only images of test patients with multiple images
        if multiple_test_image_patients_only:
            unq, counts = np.unique(test_df_split.subject.values, return_counts=True)
            pat_ids_multi = unq[counts > 1]
            test_df_split = test_df_split[test_df_split.subject.isin(pat_ids_multi)]

        test_set_representations = representation_df.representations.values[
            np.nonzero(test_df_split.image_id.values[:, None] == representation_df.img_name.values)[1]]
        test_set_patients = test_df_split.subject.values

        test_synd_ids = np.array([sid for sid in test_df_split.label])

        accs_dicts.append(
            eval_all(
                gallery_df_split,
                gallery_set_representations,
                test_set_representations,
                test_synd_ids,
                test_set_patients
            )
        )
    accs_dict_rr = mean_accs(accs_dicts)
    print_util(accs_dict_rr)

## GestaltMatcher test: Frequent, gallery: Frequent+Rare
accs_dict_fa = {}
if args.eval_split == 'all' or args.eval_split == 'fa':
    print("FREQ->ALL")
    gallery_df1 = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_gallery_images_{version}.csv'))
    gallery_df2 = pd.read_csv(os.path.join(data_path, f'gmdb_rare_gallery_images_{version}.csv'))
    gallery_df = pd.concat([gallery_df1, gallery_df2])
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

    test_df = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_test_images_{version}.csv'))

    # keep only images of test patients with multiple images
    if multiple_test_image_patients_only:
        unq, counts = np.unique(test_df.subject.values, return_counts=True)
        pat_ids_multi = unq[counts > 1]
        test_df = test_df[test_df.subject.isin(pat_ids_multi)]

    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df.image_id.values[:, None] == representation_df.img_name.values)[1]]
    test_set_patients = test_df.subject.values

    test_synd_ids = np.array([sid for sid in test_df.label])

    accs_dicts = []
    num_splits = max(gallery_df.dropna().split) + 1
    for test_split in range(int(num_splits)):
        # ids in look up table ..:
        gallery_df_split = gallery_df.fillna(test_split)
        gallery_df_split = gallery_df_split[gallery_df_split.split == test_split].reset_index()

        # Get the representations of the relevant sets
        gallery_set_representations = representation_df.representations.values[
            np.nonzero(
                gallery_df_split[gallery_df_split.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
                1]]

        accs_dicts.append(
            eval_all(
                gallery_df_split,
                gallery_set_representations,
                test_set_representations,
                test_synd_ids,
                test_set_patients
            )
        )
    accs_dict_fa = mean_accs(accs_dicts)

    print_util(accs_dict_fa)

## GestaltMatcher test: Rare, gallery: Frequent+Rare
accs_dict_ra = {}
if args.eval_split == 'all' or args.eval_split == 'ra':
    print("RARE->ALL")
    gallery_df1 = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_gallery_images_{version}.csv'))
    gallery_df2 = pd.read_csv(os.path.join(data_path, f'gmdb_rare_gallery_images_{version}.csv'))
    gallery_df = pd.concat([gallery_df1, gallery_df2])
    gallery_df['synd_id'] = np.array([sid for sid in gallery_df.label])

    test_df = pd.read_csv(os.path.join(data_path, f'gmdb_rare_test_images_{version}.csv'))

    accs_dicts = []
    num_splits = max(gallery_df.dropna().split) + 1
    for test_split in range(int(num_splits)):
        # ids in look up table ..:
        gallery_df_split = gallery_df.fillna(test_split)
        gallery_df_split = gallery_df_split[gallery_df_split.split == test_split].reset_index()
        test_df_split = test_df[test_df.split == test_split]

        # Get the representations of the relevant sets
        gallery_set_representations = representation_df.representations.values[
            np.nonzero(
                gallery_df_split[gallery_df_split.split == test_split].image_id.values[:, None] == representation_df.img_name.values)[
                1]]

        # keep only images of test patients with multiple images
        if multiple_test_image_patients_only:
            unq, counts = np.unique(test_df_split.subject.values, return_counts=True)
            pat_ids_multi = unq[counts > 1]
            test_df_split = test_df_split[test_df_split.subject.isin(pat_ids_multi)]

        test_set_representations = representation_df.representations.values[
            np.nonzero(test_df_split.image_id.values[:, None] == representation_df.img_name.values)[1]]
        test_set_patients = test_df_split.subject.values

        test_synd_ids = np.array([sid for sid in test_df_split.label])

        accs_dicts.append(
            eval_all(
                gallery_df_split,
                gallery_set_representations,
                test_set_representations,
                test_synd_ids,
                test_set_patients
            )
        )
    accs_dict_ra = mean_accs(accs_dicts)

    print_util(accs_dict_ra)
