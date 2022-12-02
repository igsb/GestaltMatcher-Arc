## lfw_similarity_matching.py
# Runs similarity matching on the LFW splits
# Can also be used to find the 'ideal' threshold

import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from lib.utils import normalize

saved_model_dir = "saved_models"


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Bone Age Test')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--model-type', default='arcface', dest='model_type',
                        help='Model type to use. (Options: FaceRecogNet, EffFaceRecogNet)')
    parser.add_argument('--in_channels', default=1, dest='in_channels')
    parser.add_argument('--num_classes', default=10575, dest='num_classes')

    # File specific
    parser.add_argument('--similarity', default='Cosine', dest='sim_type',
                        help='Type of similarity scoring/matching to use')
    parser.add_argument('--dataset', default='LFW', dest='dataset',
                        help='Dataset to use to match similarity on')
    parser.add_argument('--is_fitting', action='store_true', dest='is_fitting', default=True,
                        help='When True will try and learn/decide the similarity threshold ... '
                             '(if assigned on the fitting split)')

    return parser.parse_args()


# Function to plot a tensor as image
# optionally waits for button press before continuing
def imshow(img_t, wait=False):
    plt.close()
    plt.imshow(img_t.permute(1, 2, 0), cmap='gray')
    if wait:
        plt.waitforbuttonpress()


# Simple preprocessing used for the input images
def preprocess(img, img_size=112, color_channels=3, aspect_ratio=False):

    img = A.resize(img, img_size, img_size)

    if color_channels == 1:
        # desired number of channels is 1, so we convert to gray
        img = A.to_gray(img)[:,:,0]

    img = ToTensorV2()(image=img)["image"]
    return normalize(img, type='arcface')


def fit(model, dataset, sim_type, device):
    if sim_type == 'Cosine':
        sim_f = nn.CosineSimilarity()
    else:
        print(f"Unknown sim_type ({sim_type} given), exiting ...")
        exit()

    if dataset == 'LFW':
        dataset_dir = '../data/LFW/'
        imgs_dir = f"{dataset_dir}lfw_cropped/"

        def to_filenames(df, same_diff):
            def to_filename(name, number, prefix_dir='', postfix_img='_crop_square'):
                if prefix_dir != '':
                    prefix_dir += '\\'
                return f"{prefix_dir}{name}\\{name}_{number:04}{postfix_img}.jpg"

            return to_filename(df[0], int(df[1])), \
                   to_filename(df[0] if same_diff == 'same' else df[2],
                               int(df[2]) if same_diff == 'same' else int(df[3]))

        # fit on view1
        pairs_file = '../data/LFW/pairs_view1.csv'
        pairs = pd.read_csv(pairs_file, names=['name', 'img1', 'img2', 'img3'], skiprows=1)
        same_diff = ['same', 'diff']
        same_diff_sim_scores = [[], []]

        for pair in pairs.values:
            filename1, filename2 = to_filenames(pair, 'same' if np.isnan(pair[3]) else 'diff')
            try:
                img1 = preprocess(cv2.imread(f"{imgs_dir}{filename1}")).unsqueeze(0).to(device)
                img2 = preprocess(cv2.imread(f"{imgs_dir}{filename2}")).unsqueeze(0).to(device)
            except FileNotFoundError:
                # print("Missing file (likely due to pruning), skipping ...")
                continue
            except AttributeError:
                # opencv version of file not found
                continue

            with torch.no_grad():
                _, img1_rep = model(img1)
                _, img2_rep = model(img2)
                sim_score = sim_f(img1_rep, img2_rep)
            same_diff_sim_scores[0 if np.isnan(pair[3]) else 1].append(sim_score.item())

        threshold_scores = []
        # explore a threshold in range [0.000, 0.001, ... 0.999, 1.0]
        for thresh in range(0, 1000):
            thresh /= 1000
            same = [True if ss >= thresh else False for ss in same_diff_sim_scores[0]]
            diff = [True if ss < thresh else False for ss in same_diff_sim_scores[1]]
            threshold_scores.append(sum(same) + sum(diff))

        threshold_scores = np.array(threshold_scores)
        best_threshold = np.argmax(threshold_scores) / 1000
        best_acc = np.max(threshold_scores) / (len(same_diff_sim_scores[0]) + len(same_diff_sim_scores[1]))
        print(f"Threshold {best_threshold} reached the highest accuracy of {best_acc}")

    return best_threshold


def test(model, dataset, sim_type, threshold, device):
    if sim_type == 'Cosine':
        sim_f = nn.CosineSimilarity()
    else:
        print(f"Unknown sim_type ({sim_type} given), exiting ...")
        exit()

    if dataset == 'LFW':
        dataset_dir = '../data/LFW/'
        imgs_dir = f"{dataset_dir}lfw_cropped/"

        def to_filenames(df, same_diff):
            def to_filename(name, number, prefix_dir='', postfix_img='_crop_square'):
                if prefix_dir != '':
                    prefix_dir += '\\'
                return f"{prefix_dir}{name}\\{name}_{number:04}{postfix_img}.jpg"

            return to_filename(df[0], int(df[1])), \
                   to_filename(df[0] if same_diff == 'same' else df[2],
                               int(df[2]) if same_diff == 'same' else int(df[3]))

        accs = []
        for i in range(0, 10):
            # test on split of view2
            pairs_file = f"../data/LFW/test_splits/view2_split_{i}.csv"
            pairs = pd.read_csv(pairs_file, names=['name', 'img1', 'img2', 'img3'], delimiter='\t')
            same_diff_sim_scores = [[], []]

            for pair in pairs.values:
                filename1, filename2 = to_filenames(pair, 'same' if np.isnan(pair[3]) else 'diff')
                try:
                    img1 = preprocess(cv2.imread(f"{imgs_dir}{filename1}")).unsqueeze(0).to(device)
                    img2 = preprocess(cv2.imread(f"{imgs_dir}{filename2}")).unsqueeze(0).to(device)
                except FileNotFoundError:
                    # print("Missing file (likely due to pruning), skipping ...")
                    continue
                except AttributeError:
                    # opencv version of file not found
                    continue

                with torch.no_grad():
                    _, img1_rep = model(img1)
                    _, img2_rep = model(img2)
                    sim_score = sim_f(img1_rep, img2_rep)
                # print(sim_score.item())
                same_diff_sim_scores[0 if np.isnan(pair[3]) else 1].append(sim_score.item())

            same = [True if ss >= threshold else False for ss in same_diff_sim_scores[0]]
            diff = [True if ss < threshold else False for ss in same_diff_sim_scores[1]]
            threshold_scores = (sum(same) + sum(diff))

            acc = threshold_scores / (len(same_diff_sim_scores[0]) + len(same_diff_sim_scores[1]))
            accs.append(acc)
            print(f"\tThreshold {threshold} had an accuracy of {acc} on split {i}")
        accs = np.array(accs)
        print(f"Threshold {threshold} had a mean accuracy of {accs.mean()} with an std of {accs.std()}")


def main():
    # Training settings
    args = parse_args()

    print("Running similarity matching on LFW splits.")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #if args.model_type == 'arcface':
    model = torch.load("saved_models/s209_cisia_r50_512d_gmdb__v1.0.3_bs128__loss_size112_3channels_last_model.pth").to(device)

    # Set to evaluation mode, we're no longer training..
    model.eval()

    threshold = -1
    # Find an 'ideal' threshold on split0
    if args.is_fitting:
        threshold = fit(model, args.dataset, args.sim_type, device=device)

    test(model, args.dataset, args.sim_type, threshold=(0.327 if threshold == -1 else threshold), device=device)


if __name__ == '__main__':
    main()
