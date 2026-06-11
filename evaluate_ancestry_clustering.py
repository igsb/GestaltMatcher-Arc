import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


# To parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Evaluate GestaltMatcher-Arc for GMDB v1.1.0')

    # Seed parameter
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    # Dataset version parameters
    parser.add_argument('--test_version', dest='test_version', default='v1.1.0',
                        help='What version of GMDB was used to obtain the encodings?')
    parser.add_argument('--train_version', dest='train_version', default='v1.1.0',
                        help='What version of GMDB was used to obtain the models?')

    # Experiment parameters
    parser.add_argument('--gallery_expansion', action='store_true', default=False,
                        help='Set this flag when interested in the gallery expansion experiment for ancestry')
    parser.add_argument('--repeat_N', type=int, default=10,
                        help='How often do we repeat the experiment?')
    parser.add_argument('--stdev', action='store_true', default=False,
                        help='Set this flag to return the standard deviations, e.g. if you want to reproduce Figure 5b')

    parser.add_argument('--overlap', action='store_true', default=False,
                        help='Set this flag when running the experiment for syndromes that occur in two ancestries; '
                             'e.g. [European, Asian]')
    parser.add_argument('--overlap_ancestry_A', dest='overlap_ancestry_A', default='European',
                        help='First ancestry to check overlap between e.g. European for [European, Asian]')
    parser.add_argument('--overlap_ancestry_B', dest='overlap_ancestry_B', default='NEEDS_TO_BE_SET',
                        help='Second ancestry to check overlap between e.g. Asian for [European, Asian]')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Set this flag to get all intermediate accuracy outputs')

    return parser.parse_args()

def eval_subtract_mean_vectors(gallery_df, gallery_set_representations, test_set_representations, test_synd_ids, test_metadata=None):
    ## Experiment: subtract mean male and female representations from those sexes

    # have to reshape the array manually due to different size repr.vec. -> [model, img, [1,dim]]
    test_set_representations = [
        np.array([test_set_representations[j][i] for j in range(len(test_set_representations))]) for i in
        range(len(test_set_representations[0]))]

    gallery_set_representations = [
        np.array([gallery_set_representations[j][i] for j in range(len(gallery_set_representations))]) for i in
        range(len(gallery_set_representations[0]))]

    # test_set_representations = np.mean(test_set_representations, axis=0)
    # gallery_set_representations = np.mean(gallery_set_representations, axis=0)

    test_set_representations = np.array(test_set_representations)
    gallery_set_representations = np.array(gallery_set_representations)

    # male_mean = np.mean(gallery_set_representations[gallery_df[gallery_df.gender == 'male'].index], axis=0)/2
    # female_mean = np.mean(gallery_set_representations[gallery_df[gallery_df.gender == 'female'].index], axis=0)/2
    # unknown_mean = np.mean(gallery_set_representations[gallery_df[gallery_df.gender == 'unknown'].index], axis=0)/2

    groups = [
        (1,2)
        # ('gender', 'male'), ('gender', 'female'), ('gender', 'unknown'),
        # ('ethnicity_category', 'African'), ('ethnicity_category', 'Asian'), ('ethnicity_category', 'European'),
        # ('ethnicity_category', 'Others'), ('ethnicity_category', 'Unknown')
              ]
    # male_mean = np.mean(gallery_set_representations[:,gallery_df[gallery_df.gender == 'male'].index], axis=1).reshape(12,1,512)
    # female_mean = np.mean(gallery_set_representations[:,gallery_df[gallery_df.gender == 'female'].index], axis=1).reshape(12,1,512)
    # unknown_mean = np.mean(gallery_set_representations[:,gallery_df[gallery_df.gender == 'unknown'].index], axis=1).reshape(12,1,512)

    test_metadata = test_metadata.reset_index()
    # test_set_representations[:, test_metadata[test_metadata.gender == 'male'].index] -= male_mean
    # test_set_representations[:, test_metadata[test_metadata.gender == 'female'].index] -= female_mean
    # test_set_representations[:, test_metadata[test_metadata.gender == 'unknown'].index] -= unknown_mean
    #
    # gallery_set_representations[:, gallery_df[gallery_df.gender == 'male'].index] -= male_mean
    # gallery_set_representations[:, gallery_df[gallery_df.gender == 'female'].index] -= female_mean
    # gallery_set_representations[:, gallery_df[gallery_df.gender == 'unknown'].index] -= unknown_mean
    for group_key, group_value in groups:
        group_mean11 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'male') & (gallery_df['age'] <= 0)].index], axis=1).reshape(12, 1, 512)
        group_mean12 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'female') & (gallery_df['age'] <= 0)].index], axis=1).reshape(12, 1, 512)
        group_mean13 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'unknown') & (gallery_df['age'] <= 0)].index], axis=1).reshape(12, 1, 512)
        group_mean21 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'male') & (gallery_df['age'] > 0) & (gallery_df['age'] <= 59)].index], axis=1).reshape(12, 1, 512)
        group_mean22 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'female') & (gallery_df['age'] > 0) & (gallery_df['age'] <= 59)].index], axis=1).reshape(12, 1, 512)
        group_mean23 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'unknown') & (gallery_df['age'] > 0) & (gallery_df['age'] <= 59)].index], axis=1).reshape(12, 1, 512)
        group_mean31 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'male') & (gallery_df['age'] > 59) & (gallery_df['age'] <= 119)].index], axis=1).reshape(12, 1, 512)
        group_mean32 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'female') & (gallery_df['age'] > 59) & (gallery_df['age'] <= 119)].index], axis=1).reshape(12, 1, 512)
        group_mean33 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'unknown') & (gallery_df['age'] > 59) & (gallery_df['age'] <= 119)].index], axis=1).reshape(12, 1, 512)
        group_mean41 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'male') & (gallery_df['age'] > 119)].index], axis=1).reshape(12, 1, 512)
        group_mean42 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'female') & (gallery_df['age'] > 119)].index], axis=1).reshape(12, 1, 512)
        group_mean43 = np.mean(gallery_set_representations[:, gallery_df[(gallery_df['gender'] == 'unknown') & (gallery_df['age'] > 119)].index], axis=1).reshape(12, 1, 512)
        for df, reps in [[gallery_df, gallery_set_representations],[test_metadata, test_set_representations]]:
            reps[:, df[(df['gender'] == 'male') & (df['age'] <= 0)].index] -= group_mean11
            reps[:, df[(df['gender'] == 'female') & (df['age'] <= 0)].index] -= group_mean12
            # reps[:, df[(df['gender'] == 'unknown') & (df['age'] <= 11)].index] -= group_mean13
            reps[:, df[(df['gender'] == 'male') & (df['age'] > 0) & (df['age'] <= 59)].index] -= group_mean21
            reps[:, df[(df['gender'] == 'female') & (df['age'] > 0) & (df['age'] <= 59)].index] -= group_mean22
            # reps[:, df[(df['gender'] == 'unknown') & (df['age'] > 11) & (df['age'] <= 59)].index] -= group_mean23
            reps[:, df[(df['gender'] == 'male') & (df['age'] > 59) & (df['age'] <= 119)].index] -= group_mean31
            reps[:, df[(df['gender'] == 'female') & (df['age'] > 59) & (df['age'] <= 119)].index] -= group_mean32
            # reps[:, df[(df['gender'] == 'unknown') & (df['age'] > 59) & (df['age'] <= 119)].index] -= group_mean33
            reps[:, df[(df['gender'] == 'male') & (df['age'] > 119)].index] -= group_mean41
            reps[:, df[(df['gender'] == 'female') & (df['age'] > 119)].index] -= group_mean42
            # reps[:, df[(df['gender'] == 'unknown') & (df['age'] > 119)].index] -= group_mean43

    # Per img, per model sorted min distance from test to gallery(index)
    dists = np.stack([pairwise_distances(test_set_representations[i], gallery_set_representations[i], 'cosine')
                      for i in range(len(test_set_representations))], axis=1)
    # dists = pairwise_distances(test_set_representations, gallery_set_representations, 'cosine')
    # ranked_dists = np.argsort(dists, axis=1)

    # Condense the model-axis to end up with 1 vote per image, rather than 1 vote per model per image
    # Note: linearly weighted vote-based system has complication
    # -> we can't increment indices as syndrome id is no longer unique ...
    # .. maybe use the gallery image id as index instead
    ranked_dists = np.argsort(np.mean(dists, axis=1), axis=1)  # average the distances over all models (BAD IDEA?)
    # ranked_synds = gallery_df.values[ranked_dists][:, :, -1]
    ranked_synds = gallery_df.synd_id.values[ranked_dists]

    # This removes all duplicate occurrences except for the first one.. for each test image
    guessed_all = np.array([ranked_synds[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                            range(len(ranked_synds))])  # Expected shape: [num_images_test, num_images_gallery]

    # Top-n performance
    corr = np.zeros(4)  # 4 because [1,5,10,30]

    # sex
    corr_sex_male = np.zeros(4)
    corr_sex_female = np.zeros(4)
    corr_sex_unknown = np.zeros(4)
    num_sex_male = len(test_metadata[test_metadata.gender == 'male'])
    num_sex_female = len(test_metadata[test_metadata.gender == 'female'])
    num_sex_unknown = len(test_metadata[test_metadata.gender == 'unknown'])

    # age
    corr_age_inv = np.zeros(4)
    corr_age_0_1y = np.zeros(4)
    corr_age_1_5y = np.zeros(4)
    corr_age_5_10y = np.zeros(4)
    corr_age_10y_plus = np.zeros(4)
    num_age_inv = len(test_metadata[test_metadata.age <= 0])
    num_age_0_1y = len(test_metadata[(test_metadata.age > 0) & (test_metadata.age <= 11)])
    num_age_1_5y = len(test_metadata[(test_metadata.age > 11) & (test_metadata.age <= 59)])
    num_age_5_10y = len(test_metadata[(test_metadata.age > 59) & (test_metadata.age <= 119)])
    num_age_10y_plus = len(test_metadata[(test_metadata.age > 119)])

    # ethnicity
    corr_eth_asian = np.zeros(4)
    corr_eth_eu = np.zeros(4)
    corr_eth_afr = np.zeros(4)
    corr_eth_other = np.zeros(4)
    corr_eth_unknown = np.zeros(4)
    num_eth_asian = len(test_metadata[test_metadata.ethnicity_category == 'Asian'])
    num_eth_eu = len(test_metadata[test_metadata.ethnicity_category == 'European'])
    num_eth_afr = len(test_metadata[test_metadata.ethnicity_category == 'African'])
    num_eth_other = len(test_metadata[test_metadata.ethnicity_category == 'Others'])
    num_eth_unknown = len(test_metadata[test_metadata.ethnicity_category == 'Unknown'])

    acc_per = []
    corr_for_each_synd = []
    num_for_each_synd = []
    for i, n in enumerate([1, 5, 10, 30]):
        for idx in range(len(test_synd_ids)):
            # guessed_all[np.sort(np.unique(guessed_all, return_index=True)[1])]
            top_n_guessed = guessed_all[idx, 0:n]
            if test_synd_ids[idx] in top_n_guessed:
                corr[i] += 1

                # Sex
                if test_metadata.iloc[idx].gender == 'male':
                    corr_sex_male[i] += 1
                elif test_metadata.iloc[idx].gender == 'female':
                    corr_sex_female[i] += 1
                elif test_metadata.iloc[idx].gender == 'unknown':
                    corr_sex_unknown[i] += 1

                # Age
                if test_metadata.iloc[idx].age <= 0:
                    corr_age_inv[i] += 1
                elif test_metadata.iloc[idx].age <= 11:
                    corr_age_0_1y[i] += 1
                elif test_metadata.iloc[idx].age <= 59:
                    corr_age_1_5y[i] += 1
                elif test_metadata.iloc[idx].age <= 119:
                    corr_age_5_10y[i] += 1
                elif test_metadata.iloc[idx].age > 119:
                    corr_age_10y_plus[i] += 1

                # Ethnicity
                if test_metadata.iloc[idx].ethnicity_category == 'European':
                    corr_eth_eu[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Asian':
                    corr_eth_asian[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'African':
                    corr_eth_afr[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Others':
                    corr_eth_other[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Unknown':
                    corr_eth_unknown[i] += 1


        corr_for_each_synd.append([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]])
                                   for tl in list(set(test_synd_ids))])
        num_for_each_synd.append([len(np.where(test_synd_ids == tl)[0])
                                  for tl in list(set(test_synd_ids))])

        # Bit cluttered.., but this calculates the top-n per syndrome accuracy
        acc_per.append(sum([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]]) / len(
            np.where(test_synd_ids == tl)[0]) for tl in list(set(test_synd_ids))]) / len(
            list(set(test_synd_ids))))

    if args.verbose:
        print(f"\tOverall performance (n={len(test_metadata)}): {corr/len(test_metadata)}")

        print(f"\tSex performance: ")
        print(f"\t\tMale (n={num_sex_male}): {corr_sex_male/num_sex_male}")
        print(f"\t\tFemale (n={num_sex_female}): {corr_sex_female/num_sex_female}")
        print(f"\t\tUnknown (n={num_sex_unknown}): {corr_sex_unknown/num_sex_unknown}")

        print(f"\tEthnicity performance: ")
        print(f"\t\tAfrican (n={num_eth_afr}): {corr_eth_afr/num_eth_afr}")
        print(f"\t\tAsian (n={num_eth_asian}): {corr_eth_asian/num_eth_asian}")
        print(f"\t\tEuropean (n={num_eth_eu}): {corr_eth_eu/num_eth_eu}")
        print(f"\t\tOthers (n={num_eth_other}): {corr_eth_other/num_eth_other}")
        print(f"\t\tUnknown (n={num_eth_unknown}): {corr_eth_unknown / num_eth_unknown}")

        print(f"\tAge performance: ")
        print(f"\t\tUnknown/ x<=0 (n={num_age_inv}): {corr_age_inv / num_age_inv}")
        print(f"\t\t0 < x < 1y (n={num_age_0_1y}): {corr_age_0_1y / num_age_0_1y}")
        print(f"\t\t1 < x <= 5y (n={num_age_1_5y}): {corr_age_1_5y / num_age_1_5y}")
        print(f"\t\t5 < x <= 10y (n={num_age_5_10y}): {corr_age_5_10y / num_age_5_10y}")
        print(f"\t\t10y < x (n={num_age_10y_plus}): {corr_age_10y_plus / num_age_10y_plus}")

    return acc_per,\
        [corr, len(test_metadata)], \
        [corr_sex_male, corr_sex_female, corr_sex_unknown], \
        [num_sex_male, num_sex_female, num_sex_unknown], \
        [corr_eth_afr, corr_eth_asian, corr_eth_eu, corr_eth_other, corr_eth_unknown], \
        [num_eth_afr, num_eth_asian, num_eth_eu, num_eth_other, num_eth_unknown], \
        [corr_age_inv, corr_age_0_1y, corr_age_1_5y, corr_age_5_10y, corr_age_10y_plus], \
        [num_age_inv, num_age_0_1y, num_age_1_5y, num_age_5_10y, num_age_10y_plus]

def eval(gallery_df, gallery_set_representations, test_set_representations, test_synd_ids, test_metadata=None):
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
    # ranked_synds = gallery_df.values[ranked_dists][:, :, -1]
    ranked_synds = gallery_df.synd_id.values[ranked_dists]

    # This removes all duplicate occurrences except for the first one.. for each test image
    guessed_all = np.array([ranked_synds[i][np.sort(np.unique(ranked_synds[i], return_index=True)[1])] for i in
                            range(len(ranked_synds))])  # Expected shape: [num_images_test, num_images_gallery]

    # Top-n performance
    corr = np.zeros(4)  # 4 because [1,5,10,30]

    # sex
    corr_sex_male = np.zeros(4)
    corr_sex_female = np.zeros(4)
    corr_sex_unknown = np.zeros(4)
    num_sex_male = len(test_metadata[test_metadata.gender == 'male'])
    num_sex_female = len(test_metadata[test_metadata.gender == 'female'])
    num_sex_unknown = len(test_metadata[test_metadata.gender == 'unknown'])

    # age
    corr_age_inv = np.zeros(4)
    corr_age_0_1y = np.zeros(4)
    corr_age_1_5y = np.zeros(4)
    corr_age_5_10y = np.zeros(4)
    corr_age_10y_plus = np.zeros(4)
    num_age_inv = len(test_metadata[test_metadata.age <= 0])
    num_age_0_1y = len(test_metadata[(test_metadata.age > 0) & (test_metadata.age <= 11)])
    num_age_1_5y = len(test_metadata[(test_metadata.age > 11) & (test_metadata.age <= 59)])
    num_age_5_10y = len(test_metadata[(test_metadata.age > 59) & (test_metadata.age <= 119)])
    num_age_10y_plus = len(test_metadata[(test_metadata.age > 119)])

    # ethnicity
    corr_eth_asian = np.zeros(4)
    corr_eth_eu = np.zeros(4)
    corr_eth_afr = np.zeros(4)
    corr_eth_other = np.zeros(4)
    corr_eth_unknown = np.zeros(4)
    num_eth_asian = len(test_metadata[test_metadata.ethnicity_category == 'Asian'])
    num_eth_eu = len(test_metadata[test_metadata.ethnicity_category == 'European'])
    num_eth_afr = len(test_metadata[test_metadata.ethnicity_category == 'African'])
    num_eth_other = len(test_metadata[test_metadata.ethnicity_category == 'Others'])
    num_eth_unknown = len(test_metadata[test_metadata.ethnicity_category == 'Unknown'])

    acc_per = []
    corr_for_each_synd = []
    num_for_each_synd = []
    for i, n in enumerate([1, 5, 10, 30]):
        for idx in range(len(test_synd_ids)):
            # guessed_all[np.sort(np.unique(guessed_all, return_index=True)[1])]
            top_n_guessed = guessed_all[idx, 0:n]
            if test_synd_ids[idx] in top_n_guessed:
                corr[i] += 1

                # Sex
                if test_metadata.iloc[idx].gender == 'male':
                    corr_sex_male[i] += 1
                elif test_metadata.iloc[idx].gender == 'female':
                    corr_sex_female[i] += 1
                elif test_metadata.iloc[idx].gender == 'unknown':
                    corr_sex_unknown[i] += 1

                # Age
                if test_metadata.iloc[idx].age <= 0:
                    corr_age_inv[i] += 1
                elif test_metadata.iloc[idx].age <= 11:
                    corr_age_0_1y[i] += 1
                elif test_metadata.iloc[idx].age <= 59:
                    corr_age_1_5y[i] += 1
                elif test_metadata.iloc[idx].age <= 119:
                    corr_age_5_10y[i] += 1
                elif test_metadata.iloc[idx].age > 119:
                    corr_age_10y_plus[i] += 1

                # Ethnicity
                if test_metadata.iloc[idx].ethnicity_category == 'European':
                    corr_eth_eu[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Asian':
                    corr_eth_asian[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'African':
                    corr_eth_afr[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Others':
                    corr_eth_other[i] += 1
                elif test_metadata.iloc[idx].ethnicity_category == 'Unknown':
                    corr_eth_unknown[i] += 1


        corr_for_each_synd.append([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]])
                                   for tl in list(set(test_synd_ids))])
        num_for_each_synd.append([len(np.where(test_synd_ids == tl)[0])
                                  for tl in list(set(test_synd_ids))])

        # Bit cluttered.., but this calculates the top-n per syndrome accuracy
        acc_per.append(sum([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]]) / len(
            np.where(test_synd_ids == tl)[0]) for tl in list(set(test_synd_ids))]) / len(
            list(set(test_synd_ids))))

    if args.verbose:
        print(f"\tOverall performance (n={len(test_metadata)}): {corr/len(test_metadata)}")

        print(f"\tSex performance: ")
        print(f"\t\tMale (n={num_sex_male}): {corr_sex_male/num_sex_male}")
        print(f"\t\tFemale (n={num_sex_female}): {corr_sex_female/num_sex_female}")
        print(f"\t\tUnknown (n={num_sex_unknown}): {corr_sex_unknown/num_sex_unknown}")

        print(f"\tEthnicity performance: ")
        print(f"\t\tAfrican (n={num_eth_afr}): {corr_eth_afr/num_eth_afr}")
        print(f"\t\tAsian (n={num_eth_asian}): {corr_eth_asian/num_eth_asian}")
        print(f"\t\tEuropean (n={num_eth_eu}): {corr_eth_eu/num_eth_eu}")
        print(f"\t\tOthers (n={num_eth_other}): {corr_eth_other/num_eth_other}")
        print(f"\t\tUnknown (n={num_eth_unknown}): {corr_eth_unknown / num_eth_unknown}")

        print(f"\tAge performance: ")
        print(f"\t\tUnknown/ x<=0 (n={num_age_inv}): {corr_age_inv / num_age_inv}")
        print(f"\t\t0 < x < 1y (n={num_age_0_1y}): {corr_age_0_1y / num_age_0_1y}")
        print(f"\t\t1 < x <= 5y (n={num_age_1_5y}): {corr_age_1_5y / num_age_1_5y}")
        print(f"\t\t5 < x <= 10y (n={num_age_5_10y}): {corr_age_5_10y / num_age_5_10y}")
        print(f"\t\t10y < x (n={num_age_10y_plus}): {corr_age_10y_plus / num_age_10y_plus}")

    return acc_per,\
        [corr, len(test_metadata)], \
        [corr_sex_male, corr_sex_female, corr_sex_unknown], \
        [num_sex_male, num_sex_female, num_sex_unknown], \
        [corr_eth_afr, corr_eth_asian, corr_eth_eu, corr_eth_other, corr_eth_unknown], \
        [num_eth_afr, num_eth_asian, num_eth_eu, num_eth_other, num_eth_unknown], \
        [corr_age_inv, corr_age_0_1y, corr_age_1_5y, corr_age_5_10y, corr_age_10y_plus], \
        [num_age_inv, num_age_0_1y, num_age_1_5y, num_age_5_10y, num_age_10y_plus]


def round_array(arr):
    return list(np.around(arr*100, 2))

def main():
    # Load parameter arguments
    global args
    args = parse_args()

    # Random seed(s)
    np.random.seed(args.seed)

    # Get syndrome id from index id
    test_version = 'v1.1.0'
    version = test_version
    train_version = 'v1.1.0'
    print(f"Trained on {train_version}, testing on {test_version}")
    with open(f'lookup_table_gmdb_{train_version}.txt', 'r') as f:
        line = f.readlines()[1]
        synd_lookup_table = np.array(json.loads(line))

    # Metadata location
    data_path = os.path.join('..', 'data', 'GestaltMatcherDB', version, 'gmdb_metadata')

    # Get metadata info
    meta_df = pd.read_csv(
        os.path.join(data_path, f'image_metadata_{test_version}.tsv'),
        delimiter='\t',
        usecols=['image_id', 'gender', 'internal_syndrome_id', 'age_year', 'age_month', 'ethnicity_category']
    )
    # add an image name similar to the one used in the representation_df and age that combines years and months
    meta_df['image_name'] = [f"{id}_aligned.jpg" for id in meta_df.image_id.values]
    meta_df['age'] = [np.nan_to_num(am, nan=0) + np.nan_to_num(ay, nan=0) * 12 if not (np.isnan(am) and np.isnan(ay))
                      else -1
                      for am, ay in zip(meta_df.age_month.values, meta_df.age_year.values)]

    # Get all predictions
    # representation_df = pd.read_csv(f"all_encodings_train_{train_version}_test_{test_version}.csv", delimiter=";")
    representation_df = pd.read_csv(f"all_encodings.csv", delimiter=";")
    representation_df = representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()

    representation_df.representations = representation_df.representations.apply(lambda x: [json.loads(i) for i in x])
    representation_df.class_conf = representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])
    representation_df.img_name = representation_df.img_name.apply(lambda x: int(x.split('_')[0]))

    # GestaltMatcher test: Frequent, gallery: Frequent
    gallery_df = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_gallery_images_{version}.csv'))
    gallery_df['synd_id'] = np.array([np.where(synd_lookup_table == sid)[0][0] for sid in gallery_df.label.values if sid in synd_lookup_table])
    # gallery_df['synd_id'] = np.array([(np.where(synd_lookup_table == sid)[0][0]) if sid in synd_lookup_table else -1 for sid in gallery_df.label.values])  #workaround

    ### Experiment: change gallery set
    gallery_df['ethnicity_category'] = meta_df.iloc[[np.where(id == meta_df.image_id.values)[0][0] for id in gallery_df.image_id.values]].ethnicity_category.values
    gallery_df['gender'] = meta_df.iloc[[np.where(id == meta_df.image_id.values)[0][0] for id in gallery_df.image_id.values]].gender.values
    gallery_df['age'] = meta_df.iloc[[np.where(id == meta_df.image_id.values)[0][0] for id in gallery_df.image_id.values]].age.values
    gallery_df_eu = gallery_df[gallery_df.ethnicity_category == 'European']
    gallery_df_other = gallery_df[gallery_df.ethnicity_category != 'European']
    patient_ids_other = np.unique(gallery_df_other.subject.values)

    # Load testing images
    test_df = pd.read_csv(os.path.join(data_path, f'gmdb_frequent_test_images_{version}.csv'))

    # In case we're running the experiment w.r.t. overlapping syndromes between two ancestries
    if args.overlap:
        temp = meta_df[meta_df.image_id.isin(test_df.image_id)]
        eu_synds = np.unique(temp[temp.ethnicity_category == args.overlap_ancestry_A].internal_syndrome_id.values)
        others_synds = np.unique(temp[temp.ethnicity_category == args.overlap_ancestry_B].internal_syndrome_id.values)
        overlap_synds = eu_synds[np.isin(eu_synds, others_synds)]
        test_df = test_df[test_df.label.isin(overlap_synds)]
        del temp

    test_synd_ids = np.array([np.where(synd_lookup_table == sid)[0][0] for sid in test_df.label])
    # test_synd_ids = np.array([np.where(synd_lookup_table == sid)[0][0] if sid in synd_lookup_table else -1 for sid in test_df.label]) #workaround
    test_set_representations = representation_df.representations.values[
        np.nonzero(test_df.image_id.values[:, None] == representation_df.img_name.values)[1]]

    test_metadata = meta_df[meta_df.image_id.isin(test_df.image_id)]

    # Parameter handling
    proportion_other_param = 1 if args.overlap else 11
    divider_proportion = 1 if args.overlap else 10
    N = 1 if args.overlap else args.repeat_N

    # this leads to just the overall performance per confounder
    if not args.gallery_expansion and not args.overlap:
        divider_proportion = 1
        proportion_other_param = 1
        N = 1

    # This is where repeat our experiment N times
    all_acc_per = []
    all_overall_acc = []
    all_corrs_sex = []
    all_nums_sex = []
    all_corrs_eth = []
    all_nums_eth = []
    all_corrs_age = []
    all_nums_age = []
    for loop_i in range(N):
        print(f"Loop #{loop_i+1}:")
        all_acc_per_prop = []
        all_overall_acc_prop = []
        all_corrs_sex_prop = []
        all_nums_sex_prop = []
        all_corrs_eth_prop = []
        all_nums_eth_prop = []
        all_corrs_age_prop = []
        all_nums_age_prop = []
        for proportion_other in range(0, proportion_other_param):
            proportion_other /= divider_proportion  # convert 0 to 0.0, 1 to 0.1, 2 to 0.2, etc.
            if not args.gallery_expansion:
                proportion_other = 1.0

            if proportion_other > 0:
                sampled_patient_ids = np.random.choice(patient_ids_other, size=int(len(patient_ids_other) * proportion_other), replace=False)
                sampled_df = gallery_df_other[gallery_df_other.subject.isin(sampled_patient_ids)]
                print(f"Experiment: All EU + {proportion_other * 100:.0f}% Other ethnicities; Sampled {len(sampled_patient_ids)} patient_ids, leading to {len(sampled_df)} images.")
                gallery_df_loop = pd.concat([gallery_df_eu, sampled_df], ignore_index=True)
            else:
                print(f"Experiment: All EU")
                gallery_df_loop = gallery_df_eu

            gallery_set_representations = representation_df.representations.values[
                np.nonzero(gallery_df_loop.image_id.values[:, None] == representation_df.img_name.values)[1]]

            # Collect all data ..
            acc_per, overall_acc, corrs_sex, nums_sex, corrs_eth, nums_eth, corrs_age, nums_age \
                = eval(gallery_df_loop, gallery_set_representations, test_set_representations, test_synd_ids, test_metadata)
            all_acc_per_prop.append(acc_per)
            all_overall_acc_prop.append(overall_acc)
            all_corrs_sex_prop.append(corrs_sex)
            all_nums_sex_prop.append(nums_sex)
            all_corrs_eth_prop.append(corrs_eth)
            all_nums_eth_prop.append(nums_eth)
            all_corrs_age_prop.append(corrs_age)
            all_nums_age_prop.append(nums_age)
        all_acc_per.append(all_acc_per_prop)
        all_overall_acc.append(all_overall_acc_prop)
        all_corrs_sex.append(all_corrs_sex_prop)
        all_nums_sex.append(all_nums_sex_prop)
        all_corrs_eth.append(all_corrs_eth_prop)
        all_nums_eth.append(all_nums_eth_prop)
        all_corrs_age.append(all_corrs_age_prop)
        all_nums_age.append(all_nums_age_prop)

    if not args.overlap:
        # print mean performance for each confounder when using the usual gallery and test sets
        print("Mean accuracy when using entire gallery set")
        num_test_cases = all_overall_acc[0][0][1]  # workaround
        all_overall_acc = np.array([[overall_acc[i][0] for i in range(len(overall_acc))] for overall_acc in all_overall_acc])  # workaround
        print(f"\tOverall performance (n={num_test_cases}): {round_array(np.mean(all_overall_acc / num_test_cases, axis=0))}")  # workaround
        # print(f"\tOverall performance (n={all_overall_acc[0][0][1]}): {round_array(np.mean(np.array(all_overall_acc)[:,-1,0] / all_overall_acc[0][0][1], axis=0))}")

        print(f"\tSex performance: ")
        print(f"\t\tMale (n={all_nums_sex[0][0][0]}): {round_array(np.mean(np.array(all_corrs_sex)[:,-1,0] / all_nums_sex[0][0][0], axis=0))}")
        print(f"\t\tFemale (n={all_nums_sex[0][0][1]}): {round_array(np.mean(np.array(all_corrs_sex)[:,-1,1] / all_nums_sex[0][0][1], axis=0))}")
        print(f"\t\tUnknown (n={all_nums_sex[0][0][2]}): {round_array(np.mean(np.array(all_corrs_sex)[:,-1,2] / all_nums_sex[0][0][2], axis=0))}")

        print(f"\tEthnicity performance: ")
        print(f"\t\tAfrican (n={all_nums_eth[0][0][0]}): {round_array(np.mean(np.array(all_corrs_eth)[:,-1,0] / all_nums_eth[0][0][0], axis=0))}")
        print(f"\t\tAsian (n={all_nums_eth[0][0][1]}): {round_array(np.mean(np.array(all_corrs_eth)[:,-1,1] / all_nums_eth[0][0][1], axis=0))}")
        print(f"\t\tEuropean (n={all_nums_eth[0][0][2]}): {round_array(np.mean(np.array(all_corrs_eth)[:,-1,2] / all_nums_eth[0][0][2], axis=0))}")
        print(f"\t\tOthers (n={all_nums_eth[0][0][3]}): {round_array(np.mean(np.array(all_corrs_eth)[:,-1,3] / all_nums_eth[0][0][3], axis=0))}")
        print(f"\t\tUnknown (n={all_nums_eth[0][0][4]}): {round_array(np.mean(np.array(all_corrs_eth)[:,-1,4] / all_nums_eth[0][0][4], axis=0))}")

        print(f"\tAge performance: ")
        print(f"\t\tUnknown/ x<=0 (n={all_nums_age[0][0][0]}): {round_array(np.mean(np.array(all_corrs_age)[:,-1,0] / all_nums_age[0][0][0], axis=0))}")
        print(f"\t\t0 < x < 1y (n={all_nums_age[0][0][1]}): {round_array(np.mean(np.array(all_corrs_age)[:,-1,1] / all_nums_age[0][0][1], axis=0))}")
        print(f"\t\t1 < x <= 5y (n={all_nums_age[0][0][2]}): {round_array(np.mean(np.array(all_corrs_age)[:,-1,2] / all_nums_age[0][0][2], axis=0))}")
        print(f"\t\t5 < x <= 10y (n={all_nums_age[0][0][3]}): {round_array(np.mean(np.array(all_corrs_age)[:,-1,3] / all_nums_age[0][0][3], axis=0))}")
        print(f"\t\t10y < x (n={all_nums_age[0][0][4]}): {round_array(np.mean(np.array(all_corrs_age)[:,-1,4] / all_nums_age[0][0][4], axis=0))}")

        print(f"\tMean accuracy (over all syndromes): {round_array(np.mean(np.array(all_acc_per)[:,-1], axis=0))}")

    if args.overlap:
        # print overlap results
        indexer = ['African', 'Asian', 'European', 'Others', 'Unknown']
        index_A = indexer.index(args.overlap_ancestry_A)
        index_B = indexer.index(args.overlap_ancestry_B)

        print(f"Accuracy on overlapping syndromes (m={len(np.unique(test_synd_ids))}) when using full gallery")
        print(f"\tPerformance on [{args.overlap_ancestry_A}, {args.overlap_ancestry_B}]")
        print(f"\t\t{args.overlap_ancestry_A} (n={all_nums_eth[0][0][index_A]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, index_A] / all_nums_eth[0][0][index_A], axis=0)))}")
        print(f"\t\t{args.overlap_ancestry_B} (n={all_nums_eth[0][0][index_B]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, index_B] / all_nums_eth[0][0][index_B], axis=0)))}")

    elif args.gallery_expansion:
        # print mean performance for ancestry groups for each +m%
        print("Mean accuracy when using EU+Other gallery at m% \in ([0, 10, 20 .., 100]).")
        print(f"\tAncestry performance: ")
        print(f"\t\tAfrican (n={all_nums_eth[0][0][0]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, 0] / all_nums_eth[0][0][0], axis=0)))}")
        print(f"\t\tAsian (n={all_nums_eth[0][0][1]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, 1] / all_nums_eth[0][0][1], axis=0)))}")
        print(f"\t\tEuropean (n={all_nums_eth[0][0][2]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, 2] / all_nums_eth[0][0][2], axis=0)))}")
        print(f"\t\tOthers (n={all_nums_eth[0][0][3]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, 3] / all_nums_eth[0][0][3], axis=0)))}")
        print(f"\t\tUnknown (n={all_nums_eth[0][0][4]}): \n{np.stack(round_array(np.mean(np.array(all_corrs_eth)[:, :, 4] / all_nums_eth[0][0][4], axis=0)))}")

        if args.stdev:
            print(f"\n\tStandard deviations: ")
            print(f"\t\tAfrican: \n{np.stack(np.std(np.array(all_corrs_eth)[:, :, 0] / all_nums_eth[0][0][0], axis=0))}")
            print(f"\t\tAsian: \n{np.stack(np.std(np.array(all_corrs_eth)[:, :, 1] / all_nums_eth[0][0][1], axis=0))}")
            print(f"\t\tEuropean: \n{np.stack(np.std(np.array(all_corrs_eth)[:, :, 2] / all_nums_eth[0][0][2], axis=0))}")
            print(f"\t\tOthers: \n{np.stack(np.std(np.array(all_corrs_eth)[:, :, 3] / all_nums_eth[0][0][3], axis=0))}")
            print(f"\t\tUnknown: \n{np.stack(np.std(np.array(all_corrs_eth)[:, :, 4] / all_nums_eth[0][0][4], axis=0))}")
    print("")

if __name__ == '__main__':
    main()
