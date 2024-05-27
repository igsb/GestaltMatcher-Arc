This branch in under construction - will be updated soon.

# GestaltMatcher-Arc
This repository contains all the code used to train and evaluate our GestaltMatcher-Arc models in our Nature Genetics 
submission paper: _GestaltMatcher Database - A global reference for facial phenotypic variability in rare human diseases_
(link follows soon).\
This repo also contains snippets of code from insightface (https://github.com/deepinsight/insightface).

In order to reproduce the results, access must be requested to the GestaltMatcher DataBase (GMDB) v1.1.0.
That can be done following this link (https://db.gestaltmatcher.org/documents) if you're affiliated with a 
medical facility or faculty.

If you're only interested in reproducing the results achieved in the paper, and already have the image encodings, 
simply run the code snippets provided in the **Evaluation**-section at the bottom of this document.

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
* GMDB_original_images_v1.1.0.tar.gz

```
cd ../data/GestaltMatcherDB
tar -xzvf GMDB_original_images_v1.1.0.tar.gz
mv GMDB_original_images_v1.1.0 gmdb_images
tar -xzvf GMDB_metadata.tar.gz
mv gmdb_metadata/* .
```

Make sure your final data structure looks as follows: \
`..\data\GestaltMatcherDB\<version>`\
`...\gmdb_crops`\
`...\gmdb_images`\
`...\gmdb_metadata`,\
where `<version>` is your version of GMDB. 

### Crop and align faces
In order to get the aligned images from `gmdb_images` yourself, you have to run the `detect_pipe.py` and `align_pipe.py` from 
https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper. \
This can be useful in case you'd like e.g., a different resolution, or alignment.\
More details are in the README of that repo. \

Most importantly, the face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from 
[Google Docs](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) with pw: fstq

If you don't have GPU, please use `--cpu` to run on cpu mode.

FaceCropper command to get relevant coordinates of faces from data directory:
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_images/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --result_type coords
```

FaceCropper command to align all faces based on the coordinates according to the ArcFace alignment used by insightface:
```
python align_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_crops/ --coords_file ../data/GestaltMatcherDB/<version>/gmdb_rot/face_coords.csv
```
Note: the alignment will require the `scikit-image` package.\
Make sure to replace the `<version>` in the paths with your GMDB version; highest version at the time of writing is v1.1.0

E.g., for GMDB v1.1.0 you should run these commands:
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/v1.1.0/gmdb_images/ --save_dir ../data/GestaltMatcherDB/v1.1.0/gmdb_rot/ --result_type coords --fill_color 0
python align_pipe.py --images_dir ../data/GestaltMatcherDB/v1.1.0/gmdb_rot/ --save_dir ../data/GestaltMatcherDB/v1.1.0/gmdb_crops/ --coords_file ../data/GestaltMatcherDB/v1.1.0/gmdb_rot/face_coords.csv
```

## Train models
The training of GestaltMatcher-Arc ensemble needs to be run twice: a) for the resnet-50 mix model, and b) for the resnet-100 model.
These also require the pretrained ArcFace models from insightface: `glint360k_r50.onnx` and `glint360k_r100.onnx` to 
be in the directory `./saved_models`. \
These models can be downloaded here: https://github.com/deepinsight/insightface/tree/master/model_zoo 

To reproduce the GestaltMatcher-Arc model ensemble by training from scratch, use:
```
python train_gm_arc.py --paper_model a --epochs 50 --session 3 --local --data_dir ../data 
python train_gm_arc.py --paper_model b --epochs 50 --session 4 --local --data_dir ../data 
```

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain these results, others have not been tested.

Using the argument `--use_tensorboard` allows you to track your models training and validation curves over time.

Training a model without GPU has not been tested.

### Pretrained models
Due to ethical reasons the pretrained models are not made available publicly. \
Once access has been granted to GMDB, the pretrained model weights can be requested as well.

## Encode photos and evaluate models
With `predict_ensemble.py` you will encode all images in `--data_dir`, which by default is set to 
`../data/GestaltMatcherDB/v1.1.0/gmdb_crops`.\
The face encodings will be saved to `all_encodings_train_v1.1.0_test_1.1.0.csv` by default.

Please make sure you have the required trained models in the `./saved_models`-directory.
You need `s3_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth`, 
`s4_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth`, and \
`glint360k_r100.onnx`

Once you've checked all the constraints, run:
```
python predict_ensemble.py
```
For the machine without GPU, please use `--no_cuda`. \
(Note: It will take longer)

## Evaluation
Using the encodings you just computed, or existing provided ones (just put them into the repository's directory), as 
input for evaluation will allow you to obtain the results listed in the paper manuscript. \
As such the file `all_encodings_train_v1.1.0_test_1.1.0.csv` should be in the base directory of this repo \
(e.g. `../GestaltMatcher-Arc/all_encodings_train_v1.1.0_test_1.1.0.csv`) \
Additionally, the lookup table obtained during training (or provided), `lookup_table_gmdb_v1.1.0.txt`, should be in the same base-directory. 

Use the following code snippets to reproduce the results: \
**Performance for each confounder/group** (Table 1)\
`python evaluate_ancestry_clustering`

**Performance per ancestral group for the gallery set expansion experiment** (Figure 5b)\
`python evaluate_ancestry_clustering --gallery_expansion --N_repeat 10`

**Performance for overlapping syndromes between ancestral groups** (Table 2)\
`python evaluate_ancestry_clustering --overlap --overlap_ancestry_B African`\
`python evaluate_ancestry_clustering --overlap --overlap_ancestry_B Asian`\
`python evaluate_ancestry_clustering --overlap --overlap_ancestry_B Others`\
`python evaluate_ancestry_clustering --overlap --overlap_ancestry_B Unknown`

**Performance training set experiments** (Figure 5a) \
Further, the training set experiment's performances can be computed with `evaluate_ancestry_classification.py`. \
This script averages the per-ancestry performance over 5 different training sets, 
using permutations of EU+EU* and EU+Others. \
To that end, you need to download the performances (`performance_s<session>_seed<seed>_['eu' or 'all_anc'].npy`) and 
results (`results_s<session>_seed<seed>_['eu' or 'all_anc'].npy`) from the supplied materials. And save those to the 
directory `./experiments_anc`. 

The results can be obtained per set (EU+EU* and EU+Others) with the following two snippets: 
```
python evaluate_ancestry_classification --set eu
python evaluate_ancestry_classification --set others
```
These results are also used in the JupyterNotebook in the `./plot_scripts`.

### Performance for each confounder
Running the first snippet will result in the following output
```
Trained on v1.1.0, testing on v1.1.0
Loop #1:
Experiment: All EU + 100% Other ethnicities; Sampled 3159 patient_ids, leading to 3883 images.

Mean accuracy when using entire gallery set
	Overall performance (n=882): [56.58, 76.08, 82.65, 90.36]
	Sex performance: 
		Male (n=419): [55.37, 74.22, 80.67, 88.78]
		Female (n=393): [55.98, 75.83, 83.21, 91.09]
		Unknown (n=70): [67.14, 88.57, 91.43, 95.71]
	Ethnicity performance: 
		African (n=29): [62.07, 82.76, 82.76, 86.21]
		Asian (n=127): [53.54, 78.74, 85.04, 89.76]
		European (n=523): [55.45, 75.14, 82.6, 90.25]
		Others (n=69): [73.91, 81.16, 81.16, 92.75]
		Unknown (n=134): [53.73, 73.13, 81.34, 91.04]
	Age performance: 
		Unknown/ x<=0 (n=412): [56.31, 76.46, 84.71, 92.23]
		0 < x < 1y (n=53): [52.83, 71.7, 79.25, 90.57]
		1 < x <= 5y (n=137): [56.2, 75.91, 81.02, 90.51]
		5 < x <= 10y (n=115): [57.39, 83.48, 86.09, 90.43]
		10y < x (n=165): [58.18, 71.52, 77.58, 85.45]
```

## Reproducing the plots
Lastly, we added a Jupyter Notebook-file to the directory `plot_scripts` in which we use the aforementioned results to 
generate Figure 5a and b.

## Contact
Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
