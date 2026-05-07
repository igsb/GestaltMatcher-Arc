import cv2
import torch
import random
import numpy as np
import pandas as pd
import albumentations as A
import torch.backends.cudnn as cudnn

from onnx2torch import convert

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

    # normalize pixel values in range [-1,1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def encode(models, device, img, flip_flag=True, gray_flag=True):
    # initialize result
    result = pd.DataFrame(columns=["img_name", "model", "flip", "gray", "class_conf", "representations"])
    if flip_flag:
        flip_modes = [False, True]
    else:
        flip_modes = [False]
    if gray_flag:
        gray_modes = [False, True]
    else:
        gray_modes = [False]
    img_name = 'input'
    with torch.no_grad():
        for idx, model in enumerate(models):
            for flip in flip_modes:
                for gray in gray_modes:
                    img_p = preprocess(img,
                                       gray=gray,
                                       flip=flip
                                       ).to(device, dtype=torch.float32)

                    _pred_rep = model(img_p)
                    if len(_pred_rep) == 1:  # type == onnx --> 1 output: pred_rep
                        pred = [0]
                    else:
                        pred, _pred_rep = _pred_rep
                        pred = pred.squeeze().tolist()

                    # its overwriting right now
                    result.loc[len(result)] = [img_name, f"m{idx}", int(flip), int(gray), pred,
                                               _pred_rep.squeeze().tolist()]

    return result


def get_models():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(42)
        torch.cuda.manual_seed(42)

    # Create model
    def get_model(weights, device='cuda'):
        if ".onnx" in weights:
            _model = convert(weights).to(device)
        elif ".pth" in weights:
            _model = torch.load(weights, map_location=device).to(device)
        else:
            raise ValueError("Unknown model format")
        print(f"Loaded model: {weights}")
        return _model

    # finetuned r100
    model1 = get_model("saved_models/s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth", device=device).eval()
    # original r100
    model2 = get_model("saved_models/s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth", device=device).eval()
    # mix
    model3 = get_model("saved_models/glint360k_r100.onnx", device=device).eval()

    _models = [model1, model2, model3]

    return _models