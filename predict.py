## predict_evaluate_cases.py
# Run the chosen models on every image in the desired directory
# and save the encodings to the file: "case_encodings.csv"
# also run the evaluation-script between the "encodings.csv" (of the gallery set)
# and the new case encodings

import argparse
from glob import glob

import cv2
import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from onnx2torch import convert

from lib.models.my_arcface import MyArcFace
import torch.nn.functional as F
import albumentations as A
import torch.backends.cudnn as cudnn


# Function to normalize an image's pixel values
# Either in range [0,1] or [0,255]
def normalize(img, type='float'):
    normalized = (img - img.min()) / (img.max() - img.min())
    if type == 'int':
        return (normalized * 255).int()

    # Else: float
    return normalized


# Preprocessing used for ArcFace input image inference
def preprocess(img, img_size=112, gray=False, flip=False):
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if gray:
        # desired number of channels is 1, so we convert to gray
        img = A.to_gray(img)
    # else: color

    if flip:
        img = A.hflip(img)
    # else: normal

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Encode aligned images using GestaltMatcher-Arc ensemble')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    parser.add_argument('--data', default=['data/test_cases_align'], dest='data', nargs='+',
                        help='Path to the data directory containing the images to run the model on.')
    parser.add_argument('--save_dir', default='data/encodings', dest='save_dir',
                        help='Path to the directory where to save the encodings to.')
    parser.add_argument('--output_name', default='all_encodings.csv', dest='output_name',
                        help='Name of the output-file where to encodings will be saved to '
                             '(ignored if \'--separate_csv\' is set). Default: \'case_encodings.csv\'')

    parser.add_argument('--weights', default='saved_models', dest='weights',
                        help='Path to the directory containing the model weights. Default: \'saved_models\'')

    parser.add_argument('--model_a_path', dest='model_a_path',
                        default='s224_glint360k_r50_512d_gmdb__v1.0.3_bs64__loss_size112_3channels_last_model.pth',
                        help='Name of the file containing the weights for model A.')
    parser.add_argument('--model_b_path', dest='model_b_path',
                        default='s221_glint360k_r100_512d_gmdb__v1.0.3_bs128__loss_size112_3channels_last_model.pth',
                        help='Name of the file containing the weights for model B.')
    parser.add_argument('--model_c_path', dest='model_c_path', default='glint360k_r100.onnx',
                        help='Name of the file containing the weights for model C.')

    parser.add_argument('--img_size', default='112', type=int,
                        help='Image size to use when inferring/predicting. Default: 112')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='When set prints each file\'s name while encoding the image.')

    parser.add_argument('--separate_outputs', action='store_true', default=False,
                        help='When set saves each img in a different csv-file named \'<img_id>_encoding.csv\' '
                             '(\'--output_name\' will be ignored).')

    parser.add_argument('--save_as_pickle', action='store_true', default=False,
                        help='When set saves the encodings (as pandas DataFrame) as *.pkl instead of *.csv.')

    return parser.parse_args()
args = parse_args()


def predict(models, device, img_paths, args):
    # When storing all encodings in a single csv-file
    if not args.separate_outputs:
        if args.save_as_pickle:
            # create DataFrame to be converted to pkl-file later
            df = pd.DataFrame(columns=["img_name", "model", "flip", "gray", "class_conf", "representations"])
        else:
            # create output csv-file
            f = open(f"{args.save_dir}/{args.output_name}", "w+")
            f.write(f"img_name;model;flip;gray;class_conf;representations\n")

    img_size = args.img_size

    tick = time.time()
    with torch.no_grad():
        for idx, img_path in enumerate(img_paths):
            img_name = img_path.split('\\')[-1]
            img_name = img_name.split('/')[-1]
            if args.verbose:
                print(f"{img_name=}")
            img = cv2.imread(f"{img_path}")

            # When creating a new output-file for each image:
            if args.separate_outputs:
                if args.save_as_pickle:
                    # create DataFrame to be converted to pkl-file later
                    df = pd.DataFrame(columns=["img_name", "model", "flip", "gray", "class_conf", "representations"])
                else:
                    # create output csv-file
                    f = open(f"{args.save_dir}/{img_name.rsplit('_', 1)[0]}_encoding.csv", "w+")
                    f.write(f"img_name;model;flip;gray;class_conf;representations\n")

            for idx, model in enumerate(models):
                for flip in [False, True]:
                    for gray in [False, True]:
                        img_p = preprocess(img,
                                           img_size,
                                           gray=gray,
                                           flip=flip,
                                           ).to(device, dtype=torch.float32)

                        pred_rep = model(img_p)
                        if len(pred_rep) == 1:  # type == onnx --> 1 output: pred_rep
                            pred = [0]
                        else:  # type == pth --> 2 outputs: pred, pred_rep
                            pred, pred_rep = pred_rep
                            pred = pred.squeeze().tolist()

                        # TODO:
                        # check if we want to normalize (pred_rep = F.normalize(pred_rep))
                        # check if we want to use half-precision: has similar or better acc. and smaller size on disk

                        if args.save_as_pickle:
                            df.loc[len(df)] = [img_name, f"m{idx}", int(flip), int(gray), pred,
                                               pred_rep.squeeze().tolist()]
                        else:  # csv-file
                            f.write(f"{img_name};m{idx};{int(flip)};{int(gray)};{pred};{pred_rep.squeeze().tolist()}\n")

            if args.separate_outputs and args.save_as_pickle:
                df.to_pickle(f"{args.save_dir}/{img_name.rsplit('_', 1)[0]}_encoding.pkl")
    toc = time.time()

    # Save DataFrame as pickle
    if not args.separate_outputs:
        if args.save_as_pickle:
            # check output-file name
            fix_ext = args.output_name.split('.')
            if len(fix_ext) == 1:
                args.output_name = f"{args.output_name}.pkl"
            else:
                args.output_name = f"{fix_ext[0]}.pkl"
            df.to_pickle(f"{args.save_dir}/{args.output_name}")
        else:  # save as csv
            f.flush()
            f.close()

    print(f"Predictions took {(toc - tick):.2f}s{'.' if len(img_paths) == 1 else f' (~{((toc - tick)/len(img_paths)):.2f}s per image).'}")
    #model.train()
    return


def main():
    # Training/cuda settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    # (Aligned) data path(s):
    aligned_img_paths = []
    if len(args.data) == 1:
        print(f"Single data source found.")
        if os.path.exists(args.data[0]):
            if os.path.isfile(args.data[0]):
                # Data dir is a single file (hopefully image...)
                aligned_img_paths = args.data
            else:
                # Data dir is a dir
                aligned_img_paths = [y for x in os.walk(args.data[0]) for y in glob(os.path.join(x[0], '*.*'))]
    else:
        print(f"{len(args.data)} data sources found.")
        for d in args.data:
            if os.path.exists(d):
                if os.path.isfile(d):
                    aligned_img_paths.append(d)
                else:
                    print("--data can only contain paths or a single directory, not a mix of both.")
                    exit()
            else:
                print(f"file/directory ({d}) in --data does not exist")
    # variables: aligned_img_paths, img_names

    # Make the save directory if it doesn't exist
    os.makedirs(f"{args.save_dir}", exist_ok=True)

    # Check if the args.output_name contains the correct extension
    output_name = args.output_name.split('.')
    if len(output_name) == 1:
        if args.save_as_pickle:
            args.output_name = f"{output_name[0]}.pkl"
        else:
            args.output_name = f"{output_name[0]}.csv"

    # Create model
    def get_model(weights, device='cuda'):
        if ".onnx" in weights:
            model = convert(weights).to(device)
        elif ".pth" in weights:
            model = torch.load(weights).to(device)
        print(f"Loaded model: {weights}")
        return model

    # Create a list of models to use in the ensemble
    models = []

    # mix
    model1 = None
    if args.model_a_path != "None":
        model1 = get_model(os.path.join(args.weights, args.model_a_path), device=device)
        model1.eval()
        models.append(model1)

    # finetuned r100
    model2 = None
    if args.model_b_path != "None":
        model2 = get_model(os.path.join(args.weights, args.model_b_path), device=device)
        model2.eval()
        models.append(model2)

    # original r100
    model3 = None
    if args.model_c_path != "None":
        model3 = get_model(os.path.join(args.weights, args.model_c_path), device=device)
        model3.eval()
        models.append(model3)

    predict(models, device, aligned_img_paths, args)


if __name__ == '__main__':
    main()
