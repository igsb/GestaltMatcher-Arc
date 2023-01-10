import glob
from datetime import datetime

import cv2
from PIL import Image
import os
import torch
import torch.nn
from cffi.setuptools_ext import execfile

import streamlit as st
import pandas as pd
import numpy as np
import argparse

from onnx2torch import convert


# TODO: to implement snippet
# weight_dir = "weights"
# @st.cache()
# def load_gm_models(use_cuda=False) -> list:
#     # Create model
#     device = "cuda" if use_cuda else "cpu"
#     def get_model(weights, device='cuda'):
#         if ".onnx" in weights:
#             model = convert(weights).to(device)
#         elif ".pth" in weights:
#             model = torch.load(weights, map_location=device).to(device)
#         print(f"Loaded model: {weights}")
#         return model
#
#     # mix
#     model1 = get_model(os.path.join(weight_dir,
#                                     "s224_glint360k_r50_512d_gmdb__v1.0.3_bs64__loss_size112_3channels_last_model.pth"),
#                        device=device)
#     # finetuned r100
#     model2 = get_model(os.path.join(weight_dir,
#                                     "s221_glint360k_r100_512d_gmdb__v1.0.3_bs128__loss_size112_3channels_last_model.pth"),
#                        device=device)
#     # original r100
#     model3 = get_model(os.path.join(weight_dir, "glint360k_r100.onnx"), device=device)
#     return [model1, model2, model3]
#

def main():
    parser = argparse.ArgumentParser(description="This app lists animals")
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Number of threads to run inference with",
    )
    parser.add_argument(
        "--use_cuda", action="store_true", help="Try to use GPU (is available)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="password for sciebo login (leave empty for no loggin)",
    )
    args = parser.parse_args()

    filenames = []

    # TODO: to implement snippet
    # if torch.cuda.is_available() and args.use_cuda:
    #     if st.checkbox("use GPU (experimental for prediction speed up)"):
    #         st.write("Inference on GPU (fast)")
    #         ensemble = load_gm_models(use_cuda=True)
    #     else:
    #         st.write("Currently running inference on CPU (slow)")
    #         ensemble = load_gm_models(use_cuda=False)
    # else:
    #     ensemble = load_gm_models(use_cuda=False)

    st.title("GestaltMatcher-Arc Demo")

    file = st.file_uploader("Upload an image of a patient (png or jpg)")

    if file:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_ext = f"{name}.jpg"
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        os.makedirs(os.path.join(".", "data", "cases"), exist_ok=True)
        filename = os.path.join(".", "data", "cases", name_ext)
        cv2.imwrite(filename, img)
        filenames.append(filename)

        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)], caption=["Input"], width=300)

    if st.button("Predict"):
        # delete all the images files to avoid duplicates, mostly...
        def del_files(files):
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))

        files = glob.glob(os.path.join(".", "data", "cases", "*"))
        files.remove(filenames[0])
        del_files(files)
        files = glob.glob(os.path.join(".", "data", "cases_align", "*"))
        del_files(files)

        if not file:
            print("No image given ...")

        else:
            with st.spinner("Cropping and aligning face ..."):
                # TODO: instead of running the python script, use it's methods here instead
                # TODO: load and cache the model used for detecting the face
                os.system('python crop_align.py --no_cuda')

            with st.spinner("Encoding face ..."):
                # TODO: instead of running the python script, use it's methods here instead
                # TODO: load and cache the model used for detecting the face
                os.system('python predict.py --no_cuda --output_name case_encodings.csv')

            with st.spinner("Evaluating ..."):
                # TODO: instead of running the python script, use it's methods here instead
                # TODO: perhaps allow the top-N to be variable
                output = os.popen('python evaluate.py --case_dir data/encodings --case_list case_encodings.csv '
                                  '--gallery_dir data/encodings --gallery_list all_encodings.csv').read()
                print(f"{output}")

            st.title(f"Predicted disorder(s):")
            # st.title(f"{age:.2f} months ({age / 12:.2f} years )")

            with st.expander("See details"):
                # st.write(
                #     stats.to_html(escape=False, float_format="{:20,.2f}".format),
                #     unsafe_allow_html=True,
                # )
                st.write(output)

        is_upload_available = True


if __name__ == "__main__":
    main()
