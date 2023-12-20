import os
import json
import time
import torch
import random
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

    
def evaluate(all_df, case_df, gallery='all'):
    # Get representations of just the gallery set
    gallery_set_representations = all_df.representations.values
    gallery_set_representations = np.stack(gallery_set_representations)

    # Get representations of just the gallery set
    case_representations = case_df.representations.values
    case_representations = np.stack([case_representations])

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
        ranked_dist_index = np.argsort(mean_dists, axis=1)  # average the distances over all models
        ranked_mean_dists = np.take_along_axis(mean_dists, ranked_dist_index, axis=1)
        ranked_img_ids = gallery_df["img_name"].values[ranked_dist_index]

        return ranked_mean_dists, ranked_img_ids

    # return ranked_mean_dists, ranked_img_ids
    return eval(all_df, gallery_set_representations, case_representations)

def get_first_synds(ranked_mean_dists_list, ranked_img_ids_list, images_synds_dict, verbose=False):
    # This removes all duplicate occurrences except for the first one.. for each test image
    img_synds_results_list = []
    img_dists_results_list = []
    img_image_results_list = []

    for ranked_img_ids, ranked_mean_dists in zip(ranked_img_ids_list, ranked_mean_dists_list):
        # the result of predicting one image
        img_synd_results = []
        img_dists = []
        img_ids = []
        for image_id, dist in zip(ranked_img_ids, ranked_mean_dists):
            image_synd = images_synds_dict[int(image_id)]

            img_synd_results.append(image_synd['disorder_internal_id'])
            img_dists.append(dist)
            img_ids.append(image_id)
        img_synd_results = np.array(img_synd_results)
        img_dists = np.array(img_dists)
        img_ids = np.array(img_ids)

        sort_index = np.sort(np.unique(img_synd_results, return_index=True)[1])
        ranked_unique_synds = np.array(img_synd_results[sort_index])
        ranked_unique_dists = np.array(img_dists[sort_index])
        ranked_unique_image_ids = np.array(img_ids[sort_index])
        img_synds_results_list.append(ranked_unique_synds)
        img_dists_results_list.append(ranked_unique_dists)
        img_image_results_list.append(ranked_unique_image_ids)
    img_synds_results_list = np.array(img_synds_results_list)
    img_dists_results_list = np.array(img_dists_results_list)
    img_image_results_list = np.array(img_image_results_list)

    return img_synds_results_list, img_dists_results_list, img_image_results_list


def get_first_genes(ranked_mean_dists_list, ranked_img_ids_list, image_genes_dict, verbose=False):
    # This removes all duplicate occurrences except for the first one.. for each test image
    img_genes_results_list = []
    img_dists_results_list = []
    img_image_results_list = []

    for ranked_img_ids, ranked_mean_dists in zip(ranked_img_ids_list, ranked_mean_dists_list):
        # the result of predicting one image
        img_gene_results = []
        img_dists = []
        img_ids = []
        for image_id, dist in zip(ranked_img_ids, ranked_mean_dists):
            genes = image_genes_dict[int(image_id)]
            for gene in genes:
                img_gene_results.append(gene['gene_internal_id'])
                img_dists.append(dist)
                img_ids.append(image_id)
        img_gene_results = np.array(img_gene_results)
        img_dists = np.array(img_dists)
        img_ids = np.array(img_ids)

        sort_index = np.sort(np.unique(img_gene_results, return_index=True)[1])
        ranked_unique_genes = np.array(img_gene_results[sort_index])
        ranked_unique_dists = np.array(img_dists[sort_index])
        ranked_unique_image_ids = np.array(img_ids[sort_index])
        img_genes_results_list.append(ranked_unique_genes)
        img_dists_results_list.append(ranked_unique_dists)
        img_image_results_list.append(ranked_unique_image_ids)
    img_genes_results_list = np.array(img_genes_results_list)
    img_dists_results_list = np.array(img_dists_results_list)
    img_image_results_list = np.array(img_image_results_list)

    return img_genes_results_list, img_dists_results_list, img_image_results_list


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


    return synds_all, dists_all, imgs_all, subjects_all


def prep_csv(df, is_pickle=False):
    df = df.groupby('img_name').agg(lambda x: list(x)).reset_index()
    if not is_pickle:
        df.representations = df.representations.apply(lambda x: np.array([json.loads(i) for i in x]))
    df.img_name = df.img_name.apply(lambda x: x.split('_')[0])
    return df

def get_encodings_set(encoding_input, encoding_list=[]):
    # Check whether use a single file or all files in the directory
    is_separate = True
    if os.path.isfile(encoding_input):
        is_separate = False

    df_main = None
    # single file
    if not is_separate:
        if os.path.splitext(encoding_input)[-1] == '.pkl':
            df_main = prep_csv(pd.read_pickle(encoding_input), is_pickle=True)
        else:  # is csv
            df_main = prep_csv(pd.read_csv(encoding_input, delimiter=';'))
    else:
        # else: separate files or dir
        encoding_files = os.listdir(encoding_input)
        for filename in encoding_files:
            names = os.path.splitext(filename)
            prefix_name = names[0]
            suffix_name = names[-1]
            if len(encoding_list) > 0 and prefix_name not in encoding_list:
                # ignore if not in the list
                continue
            if suffix_name == '.pkl':
                df_part = prep_csv(pd.read_pickle(os.path.join(encoding_input, filename)), is_pickle=True)
            else:  # is csv
                df_part = prep_csv(pd.read_csv(os.path.join(encoding_input, filename), delimiter=';'))
            df_main = pd.concat([df_main, df_part])
    return df_main


def print_format_output(results):
    synd_ids = results[0][0]
    dists = results[1][0]
    img_ids = results[2][0]
    subject_ids = results[3][0]
    print(f"Synd ids: {list(synd_ids)}")
    print(f"Distances: {[round(i, 3) for i in dists]}")
    print(f"Image ids: {[int(i) for i in img_ids]}")
    print(f"Subject ids: {list(subject_ids)}")


def format_syndrome_json(results, synds_metadata_dict, images_dict, case_id=''):
    synd_ids = results[0][0]
    dists = results[1][0]
    img_ids = results[2][0]

    output_list = []
    for synd_id, dist, image_id in zip(synd_ids, dists, img_ids):
        output = {'syndrome_name': synds_metadata_dict[int(synd_id)]['disorder_name'],
                  'omim_id': synds_metadata_dict[int(synd_id)]['omim_id'],
                  'distance': round(float(dist), 3),
                  'gestalt_score': round(float(dist), 3),
                  'image_id': image_id,
                  'subject_id': str(images_dict[int(image_id)]['patient_id'])}
        output_list.append(output)

    return output_list


def format_gene_json(results, genes_metadata_dict, images_dict, case_id=''):
    genes = results[0][0]
    dists = results[1][0]
    img_ids = results[2][0]

    output_list = []
    for gene, dist, image_id in zip(genes, dists, img_ids):
        output = {'gene_name': genes_metadata_dict[int(gene)]['gene_name'],
                  'gene_entrez_id': genes_metadata_dict[int(gene)]['gene_entrez_id'],
                  'distance': round(float(dist), 3),
                  'gestalt_score': round(float(dist), 3),
                  'image_id': image_id,
                  'subject_id': str(images_dict[int(image_id)][0]['patient_id'])}
        output_list.append(output)

    return output_list


def save_to_json(results, output_dir, output_file):
    output_filename = os.path.join(output_dir, output_file)
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def get_gallery_encodings_set(images_synds_dict):
    gallery_list = []
    gallery_input = os.path.join('data', 'gallery_encodings', 'GMDB_gallery_encodings_10102023_v1.0.7.pkl')
    gallery_df = get_encodings_set(gallery_input, gallery_list)
    image_ids = [str(i) for i in images_synds_dict.keys()]
    gallery_df = gallery_df[gallery_df["img_name"].isin(image_ids)]
    print("Load gallery encodings")
    return gallery_df


def predict(test_df, _gallery_df, images_synds_dict, images_genes_dict, genes_metadata, synds_metadata):
    start_time = time.time()
    # Seed everything
    np.random.seed(1000)
    torch.manual_seed(1000)
    random.seed(1000)

    ### set input case file
    case_df = test_df
    parse_finished_time = time.time()

    ## Evaluate
    # Get all synd_ids, dists, img_ids, subject_ids per image in gallery
    top_n = 'all'
    if top_n == 'all':
        n = None
    else:
        n = int(args.top_n)

    all_ranks = evaluate(_gallery_df, case_df, "all")
    all_ranks = np.array(all_ranks)

    evaluate_finished_time = time.time()

    # Get all synd_ids, dists, img_ids, subject_ids per syndrome in gallery
    first_synd_ranks = get_first_synds(*all_ranks, images_synds_dict)
    first_synd_ranks = np.array(first_synd_ranks)
    get_synds_time = time.time()

    # Get all synd_ids, dists, img_ids, subject_ids per syndrome in gallery
    first_gene_ranks = get_first_genes(*all_ranks, images_genes_dict)
    first_gene_ranks = np.array(first_gene_ranks)
    get_genes_time = time.time()
    case_id = 1

    synd_output_list = format_syndrome_json(first_synd_ranks[:, :, :n], synds_metadata, images_synds_dict, case_id)
    gene_output_list = format_gene_json(first_gene_ranks[:, :, :n], genes_metadata, images_genes_dict, case_id)

    output_finished_time = time.time()

    #print('Parse: {:.2f}s'.format(parse_finished_time-start_time))
    print('Evaluate: {:.2f}s'.format(evaluate_finished_time-parse_finished_time))
    #print('Get synds: {:.2f}s'.format(get_synds_time-evaluate_finished_time))
    #print('Get genes: {:.2f}s'.format(get_genes_time-get_synds_time))
    #print('Format: {:.2f}s'.format(output_finished_time-get_genes_time))
    #print('Total: {:.2f}s'.format(output_finished_time-start_time))
    output = {"model_version": "v1.0.7",
              "gallery_version": "10.10.2023",
              "suggested_genes_list": gene_output_list,
              "suggested_syndromes_list": synd_output_list}
    return output

