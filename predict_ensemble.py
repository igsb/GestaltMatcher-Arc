## predict_ensemble.py
# Run the chosen model on every image in the desired data directory
# and save the encodings to the file: "all_encodings.csv"

import argparse
import cv2
import datetime
import os
import random

import torch
import numpy as np
from onnx2torch import convert
import torch.nn.functional as F
import albumentations as A

from lib.models.my_arcface import MyArcFace

saved_model_dir = "saved_models"


# Preprocessing used for ArcFace input image inference
def preprocess(img, img_size=112, gray=False, flip=False):
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if gray:
        # desired number of channels is 1, so we convert to gray
        img = A.to_gray(img)
    #else: color

    if flip:
        img = A.hflip(img)
    #else: normal

    # normalize pixel values in range [-1,1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Predict GestaltMatcher-Arc Ensemble')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--data_dir', default='../data/GestaltMatcherDB/v1.0.3/gmdb_align', dest='data_dir',
                        help='Path to the data directory containing the images to run the model on.')
    parser.add_argument('--weight_dir', default='../data/GestaltMatcherDB/v1.0.3/gmdb_align', dest='weight_dir',
                        help='Path to the directory containing the model weights.')

    return parser.parse_args()


def predict(models, device, data, args):
    for model in models:
        model.eval()

    f = open("all_encodings.csv", "w+")
    f.write(f"img_name;model;flip;gray;class_conf;representations\n")

    tick = datetime.datetime.now()
    with torch.no_grad():
        for idx, img_path in enumerate(data):
            print(f"{img_path=}")
            img = cv2.imread(os.path.join(args.data_dir, img_path))

            for idx, model in enumerate(models):
                for flip in [False,True]:
                    for gray in [False,True]:
                        img_p = preprocess(img,
                                           gray=gray,
                                           flip=flip
                                           ).to(device, dtype=torch.float32)

                        pred_rep = model(img_p)
                        if len(pred_rep) == 1:  #type == onnx --> 1 output: pred_rep
                            pred = [0]
                        else:
                            pred, pred_rep = pred_rep
                            pred = pred.squeeze().tolist()

                        pred_rep = F.normalize(pred_rep)
                        f.write(f"{img_path};m{idx};{int(flip)};{int(gray)};"
                                f"{pred};{pred_rep.squeeze().tolist()}\n")

    f.flush()
    f.close()

    print(f"Predictions took {datetime.datetime.now() - tick}s")
    return


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    data = os.listdir(args.data_dir)

    # Create model
    def get_model(weights, device='cuda'):
        if ".onnx" in weights:
            model = convert(weights).to(device)
        elif ".pth" in weights:
            model = torch.load(weights, map_location=device).to(device)
        print(f"Loaded model: {weights}")
        return model

    # mix
    model1 = get_model(os.path.join(args.weight_dir,
                                    "s224_glint360k_r50_512d_gmdb__v1.0.3_bs64__loss_size112_3channels_last_model.pth"),
                       device=device)
    # finetuned r100
    model2 = get_model(os.path.join(args.weight_dir,
                                    "s221_glint360k_r100_512d_gmdb__v1.0.3_bs128__loss_size112_3channels_last_model.pth"),
                       device=device)
    # original r100
    model3 = get_model(os.path.join(args.weight_dir, "glint360k_r100.onnx"), device=device)

    models = [model1, model2, model3]
    predict(models, device, data, args)


if __name__ == '__main__':
    main()
