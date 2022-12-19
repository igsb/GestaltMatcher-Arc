# GestaltMatcher-Arc
This repository contains all the code used to train and evaluate our GestaltMatcher-Arc models in our WACV2023 
accepted paper: Improving Deep Facial Phenotyping for Ultra-rare Disorder Verification Using Model Ensembles 
(https://arxiv.org/abs/2211.06764 ).\
This repo also contains snippets of code from insightface (https://github.com/deepinsight/insightface).

In order to reproduce the results access must be requested to the GestaltMatcher DataBase (GMDB).
That can be done following this link (https://db.gestaltmatcher.org/documents) if you're affiliated with a 
medical facility or faculty.

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

## Data preparation
The data should be stored in `../data/GestaltMatcherDB/<version>`, it can be downloaded from http://gestaltmatcher.org 
on request. \
Please download the following two files from GMDB website:
* GMDB metadata
* GMDB_original_images_v1.0.3.tar.gz

```
cd ../data/GestaltMatcherDB
tar -xzvf GMDB_original_images_v1.0.3.tar.gz
mv GMDB_original_images_v1.0.3 gmdb_images
tar -xzvf GMDB_metadata.tar.gz
mv gmdb_metadata/* .
```

Make sure your final data structure looks as follows: \
`..\data\GestaltMatcherDB\<version>`\
`...\gmdb_images`\
`...\gmdb_metadata`,\
where `<version>` is your version of GMDB. 

### Crop and align faces
In order to get the aligned images, you have to run the `detect_pipe.py` and `align_pipe.py` from 
https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper. \
More details are in the README of that repo. \
Most importantly the face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from 
[Google Docs](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) with pw: fstq

The face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from the repository 
mentioned above.\
If you don't have GPU, please use `--cpu` to run on cpu mode.

FaceCropper command to get relevant coordinates of faces from data directory:
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_images/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --result_type coords
```

FaceCropper command to align all faces based on the coordinates according to the ArcFace alignment used by insightface:
```
python align_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_align/ --coords_file ../data/GestaltMatcherDB/<version>/gmdb_rot/face_coords.csv
```
Note: the alignment will require the `scikit-image` package.\
Make sure to replace the `<version>` in the paths with your GMDB version; highest version at the time of writing is v1.0.3

## Train models
The training of GestaltMatcher-Arc needs to be run twice: a) for the resnet-50 mix model, and b) for the resnet-100 model.
For these also require the pretrained ArcFace models from insightface: `glint360k_r50.onnx` and `glint360k_r100.onnx` to 
be in the directory `./saved_models`. \
These models can be downloaded here: https://github.com/deepinsight/insightface/tree/master/model_zoo 

To reproduce our Gestalt Matcher model listed in the table by training from scratch, use:
```
python train_gm_arc.py --paper_model a --epochs 50 --session 1 --dataset gmdb --in_channels 3 --img_size 112 --use_tensorboard --local --data_dir ../data 
python train_gm_arc.py --paper_model b --epochs 50 --session 2 --dataset gmdb --in_channels 3 --img_size 112 --use_tensorboard --local --data_dir ../data 
```

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain these results, others have not been tested.

Using the argument `--use_tensorboard` allows you to track your models training and validation curves over time.

Training a model without GPU has not been tested.

### Pretrained models
Due to ethical reasons the pretrained models are not made available publicly. \
Once access has been granted to GMDB, the pretrained model weights can be requested as well.

## Encode photos and evaluate models
With `python predict_ensemble.py` you will encode all images in `--data_dir`, which by default is set to 
`../data/GestaltMatcherDB/v1.0.3/gmdb_align`.\
The face encodings will be saved to `all_encodings.csv`.

For the machine without GPU, please use `--no_cuda`.

The following command will generate `all_encodings.csv` using the three models in our model ensemble, as well as the 
test time augmentation described in the paper:

```
python predict_ensemble.py
```

### Evaluation
Using the previously computed encodings as input for evaluation will allow you to obtain the results listed in the table.

```
python evaluate_ensemble.py

===========================================================
---------   test: Frequent, gallery: Frequent    ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-frequent|5761    |593   |52.99 |71.01 |79.19 |89.99 |
---------       test: Rare, gallery: Rare        ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |792.7   |312.3 |35.98 |53.93 |62.43 |76.56 |
--------- test: Frequent, gallery: Frequent+Rare ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-frequent|6553.7  |593   |50.79 |69.17 |76.66 |88.37 |
---------   test: Rare, gallery: Frequent+Rare   ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |6553.7  |312.3 |24.05 |38.44 |44.53 |57.95 |
===========================================================

```


## Contact
Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
