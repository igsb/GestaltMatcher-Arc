# GestaltMatcher-Arc: Service
This is the `service`-branch of the repository, containing all the code for our GestaltMatcher service.
This repo also contains snippets of code from insightface (https://github.com/deepinsight/insightface); both from their 
alignment process and their RetinaFace detector.

The concept is to first acquire the aligned face, followed by encoding it, and lastly comparing the encoding to a set
of gallery encodings. The expected in- and output of each stage is described after `Environment`-section.

## Environment
Please use python version 3.7+, and the package listed in requirements.txt.

```
python3 -m venv env_gm
source env_gm/Scripts/activate
pip install -r requirements.txt
```

If you would like to train and evaluate with GPU, please remember to install cuda in your system.
If you don't have GPU, please choose the CPU option (`--no_cuda`) in the following section.

Follow these instructions (https://developer.nvidia.com/cuda-downloads ) to properly install CUDA.
Follow the necessary instructions (https://pytorch.org/get-started/locally/ ) to properly install PyTorch, you might still need additional dependencies (e.g. Numpy).
Using the following command should work for most using the `conda` virtual env.
```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```

If any problems occur when installing the packages in `requirements.txt`, the most important packages are:
```
numpy
pandas
pytorch=1.9.0
torchvision=0.10.0
tensorboard
opencv
matplotlib
scikit-image
scikit-learn
onnx2torch
albumentations
```

## Crop and align faces
In order to get aligned images, you have to run `crop_align.py`. It is possible to either crop and align a single image,
multiple images in a list or a directory of images.\
With `python crop_align.py` you will crop and align all images in `--data` (default: `./data/cases`) and save them to 
the `--save_dir` (default: `./data/cases_align`). This is quite free-form and does not need to be a directory, but can 
also be an image name or list of image names.

The face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from 
[Google Docs](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) with pw: fstq \
If you don't have GPU, please use `--no_cuda` to run on cpu mode.

## Encode photos 
With `python predict.py` you will encode all images in `--data` (default: `./data/cases_align`). This is quite free-form
and does not need to be a directory, but can also be an image name or list of image names. \
There are several options w.r.t. saving the encodings. By default, the encoding for each image is saved into a single 
`*.csv`-file named by `--output_name` (default: `all_encodings.csv`) which is stored in the directory given by 
`--save_dir` (default: `data/encodings`).\
Alternatively, you can choose to save all encodings into separate files, holding only the encodings per image, using 
`--separate_outputs`. In this case the files will be named after the image name and those outputs will be saved in the 
`--save_dir`.\
Lastly, it is possible to save the encodings directly as a pickle of a DataFrame. In this case you should use the flag 
`--save_as_pickle`. The `--output_name` then end in `*.pkl` instead. 

For machines without a GPU, please use `--no_cuda`.

### Pretrained models
Due to ethical reasons the pretrained models are not made available publicly. \
Once access has been granted to GMDB, the pretrained model weights can be requested as well.

The pretrained models by default are stored in a directory set by `--weight_dir` (default:`./saved_models/`). Further, 
using the arguments `--model_a_path`, `--model_b_path` and `--model_c_path`, the paths within this directory need to be
specified (default: uses all supplied model names). \
When setting any of those to 'None' they will not be included in the ensemble.

## Evaluate encodings with gallery encodings
With `evaluate.py` you can evaluate case encodings using gallery encodings.

There are several ways to load the encodings, either using a single file containing all encodings, or separate encoding-
files (e.g. after using `--seperate_outputs` for `predict.py`) for each image. \
Use `--case_input` to specify the test encoding. You can use single file or a folder as input.
When you specify the folder, we will parse all the files in the folder as test images.
For the gallery, please specify with `--gallery_input`, you can use single file or a folder as input.
When you specify the folder, we will parse all the files in the folder as gallery images.
You can further use `--gallery_list_file` to specify the image name you want to include in the gallery.
If you'd rather use only two file, for all gallery encodings and all case encodings, you simply specify only those
filenames within their respective directories (and do NOT use `--separate_files_<x>`).

Next, you need to specify the directory containing the GMDB metadata using `--metadata_dir`.

For the output format, you can use `--top_n` to choose the number of entries in the output file.
If you choose `--top_n all`, it will output all the syndromes/genes in the gallery.

Lastly, you will need to specify the lookup table that was used during the model training, which is automatically 
generated and saved when running the training. However, it is included in the directory under the name 
`lookup_table_gmdb.txt` and is the default path of `--lut` (the argument used to set it).

For the output file, please indicate the directory with `--output_dir` and the output filename with `--output_file`.

## Streamlit application
We offer a very basic app created with `streamlit`.\
To run it you will need to install the following package first: `pip install streamlit`.\
Afterwards enter this into your terminal: `streamlit run streamlit.py`. 
Thereafter, you can access it within you localhost (by default on port 8501).

It can be used by simply inputting a photo and with a click on the button will output the top-5 disorders as well as 
the top-5 most similar matches, most similar disorders, and most similar patients, including their `syndrome_id`, 
`distance`, `img_id`, and `patient_id`, respectively.\
In order for this to work you will need the trained models, and the gallery encodings in 
`./data/encodings/all_encodings.csv`. 

It is currently not optimized. However, addressing following TODO's will allow it to run more smoothly:
- [ ] Replace the `os.system`/`os.popen`-calls with functions from those python scripts
- [ ] Cache the models and the gallery encodings
- [ ] Consider multi-threading or using GPU


## Contact
Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
