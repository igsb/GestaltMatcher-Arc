# GestaltMatcher-Arc: Service
This repository contains all the code for our GestaltMatcher service and the training of GestaltMatcher-Arc.
For more details, please check our publications in the [references section](#references).
This repo also contains snippets of code from insightface (https://github.com/deepinsight/insightface); both from their 
alignment process and their RetinaFace detector.

<p><strong>There are two parts in this repo:

1. [GestaltMatcher REST api](#gestaltmatcher-rest-api) to run the prediction. You don't have to train the models.
1. [Step-by-step setup](#step-by-step-setup), from preprocessing, training to evaluation.</strong></p>

The concept is to first acquire the aligned face, followed by encoding it, and lastly comparing the encoding to a set
of gallery encodings. The expected in- and output of each stage is described after `Environment`-section.

To simplify the process, we setup the GestaltMatcher REST api. You can build the docker image and host
the service via REST api. Then, you can obtain the prediction results by sending the image through
REST api. You can find more detail in [GestaltMatcher REST api](#gestaltmatcher-rest-api)

## GestaltMatcher REST api
We can host GestaltMatcher as a service via REST api. You can build the Docker image and host GM service in your local machine.
### Requirements
Please contact us to obtain the following files and store them in the corresponding paths.

**Pretrained model**

Save the following files in ./saved_models/
1. Resnet50_Final.pth (for the face alignment)
2. s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth (model 1 for the encoding)
3. s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth (model 2 for the encoding)
4. glint360k_r100.onnx (model 3 for the encoding)

**Metadata**

Save the following file in ./data/
1. image_gene_and_syndrome_metadata_20082024.p (image metadata)

**Encodings**

Save the following file in ./data/gallery_encodings/
1. GMDB_gallery_encodings_20082024_v1.1.0_service.pkl (image encodings)

**Config file**

config.json stores the username and password required for the authentication. Please change the default setting before
starting the REST API. Please also change the username and password in send_image_api.py. 

### Build and run docker image
Build docker image: `docker build -t gm-api .`

Run and listen the request in localhost:5000:`docker run -p 5000:5000 gm-api`

### Main flow
Please check the main.py for the main flow of the analysis from loading model, cropping and evaluating.
```
# take the predict endpoint for example
async def predict_endpoint(username: Annotated[str, Depends(get_current_username)], image: Img):
    img = readb64(image.img)

    aligned_img = face_align_crop(_cropper_model, img, _device)

    encoding = encode(_models, 'cpu', aligned_img, True, True)

    result = predict(encoding,
                     _gallery_df,
                     _images_synds_dict,
                     _images_genes_dict,
                     _genes_metadata_dict,
                     _synds_metadata_dict)

```

### Send request
You can send a single image or multiple images in a folder to the api via **send_image_api.py**.
```
python send_image_api.py --case_input demo_images/cdls_demo.png --output_dir output

# arguments:
--case_input :input single file or dir containing multiple images
--output_dir :output dir
--url :url for the service, default: localhost
--port :port for the service, default: 5000
```

### Results
The results will be saved in a file with JSON format. There are three information stored in the file.
1. case_id: the original filename without file extension
2. suggested_genes_list: the ranked gene list sorted by the distance
3. suggested_syndromes_list: the ranked syndrome list sorted by the distance

#### suggested_genes_list (for variants prioritization)
A gene list sorted by the distance in ascending order which can be used for variant prioritization.
* **distance** is the cosine distance to the nearest image with the gene in the gallery. A smaller distance indicates a higher similarity.
* **image_id** is the image_id in GestaltMatcher Database which is the nearest image of that gene in the gallery.
* **subject_id** is the patient_id in GestaltMatcher Database which is the nearest patient of that gene in the gallery.
* **gene_entrez_id and gene_name** the gene id and gene name.
* **gestalt score** is the same as the distance.

**Note:** some syndromes have no gene associated because they are the chromosomal abnormality or huge deletion that cover
multiple genes. We still keep them in the entry. For example, WILLIAMS-BEUREN SYNDROME; WBS has no gene associated in OMIM, so we use gene_name: WILLIAMS-BEUREN SYNDROME; WBS and gene_entrez_id: null for this entry.
Please filter out this kind of entry with null gene_entrez_id if you do need them.  

```angular2html
{    
    "case_id": "cdls_demo",
    "model_version": "v1.0.3",
    "gallery_version": "v1.0.3",
    "suggested_genes_list": [
        {
            "gene_name": "NIPBL",
            "gene_entrez_id": "25836",
            "distance": 0.44,
            "gestalt_score": 0.44,
            "image_id": "4883",
            "subject_id": "3546"
        },
        {
            "gene_name": "SMC1A",
            "gene_entrez_id": "8243",
            "distance": 0.516,
            "gestalt_score": 0.516,
            "image_id": "8513",
            "subject_id": "5656"
        },
        {
            "gene_name": "HDAC8",
            "gene_entrez_id": "55869",
            "distance": 0.516,
            "gestalt_score": 0.516,
            "image_id": "8513",
            "subject_id": "5656"
        },...
    ],
    "suggested_genes_list": [...]
}
```


#### suggested_syndromes_list
A syndrome list sorted by the distance in ascending order.
* **distance** is the cosine distance to the nearest image with the gene in the gallery. A smaller distance indicates a higher similarity.
* **image_id** is the image_id in GestaltMatcher Database which is the nearest image of that gene in the gallery.
* **subject_id** is the patient_id in GestaltMatcher Database which is the nearest patient of that gene in the gallery.
* **syndrome_name and omim_id** the syndrome name and omim id.
* **gestalt score** is the same as the distance.
```angular2html
    "suggested_syndromes_list": [
        {
            "syndrome_name": "Cornelia de Lange syndrome",
            "omim_id": 122470,
            "distance": 0.44,
            "gestalt_score": 0.44,
            "image_id": "4883",
            "subject_id": "3546"
        },
        {
            "syndrome_name": "DDX23",
            "omim_id": "",
            "distance": 0.575,
            "gestalt_score": 0.575,
            "image_id": "8998",
            "subject_id": "5949"
        },
        {
            "syndrome_name": "SMITH-MAGENIS SYNDROME; SMS",
            "omim_id": 182290,
            "distance": 0.699,
            "gestalt_score": 0.699,
            "image_id": "5961",
            "subject_id": "4239"
        },...
    ]
```
## Step-by-step setup
### Environment
Please use python version 3.8 or (3.7+), and the package listed in requirements.txt.

<p><strong>The following setup is verified in the following environments:

* Window 11
* RTX4090
* cuda 11.8
* pytorch 2.3.1</strong></p>

```
python3 -m venv env_gm
source env_gm/Scripts/activate
pip install -r requirements.txt
```

If you would like to train and evaluate with GPU, please remember to install cuda in your system.
If you don't have GPU, please choose the CPU option (`--no_cuda`) in the following section.

Follow these instructions (https://developer.nvidia.com/cuda-downloads) to properly install CUDA.

Follow the necessary instructions (https://pytorch.org/get-started/locally/) to properly install PyTorch, you might still need additional dependencies (e.g. Numpy).
Using the following command should work for most using the `conda` virtual env.
```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

If any problems occur when installing the packages in `requirements.txt`, the most important packages are:
```
numpy==1.24.4
pandas==2.0.3
torch==2.3.1
torchaudio==2.3.1
torchvision==0.18.1
tensorboard==2.14.0
opencv-python-headless==4.10.0.84
matplotlib==3.7.5
scikit-image==0.21.0
scikit-learn==1.3.2
onnx==1.16.1
onnx2torch==1.4.1
albumentations==1.2.1
pillow==10.4.0
```

### Pretrained models
Due to ethical reasons the pretrained models are not made available publicly. \
Once access has been granted to GMDB, the pretrained model weights can be requested as well.

Save the following files in ./saved_models/
1. Resnet50_Final.pth (for the face alignment)
2. glint360k_r50.onnx (base pre-trained model for model a)
3. glint360k_r100.onnx (base pre-trained model for model b)

Trained models by our group:
1. s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth (model 1 for the encoding)
2. s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth (model 2 for the encoding)

### Crop and align faces
In order to get aligned images, you have to run `crop_align.py`. It is possible to either crop and align a single image,
multiple images in a list or a directory of images.\
With `python crop_align.py` you will crop and align all images in `--data` (default: `./data/cases`) and save them to 
the `--save_dir` (default: `./data/cases_align`). This is quite free-form and does not need to be a directory, but can 
also be an image name or list of image names.

The face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from 
[Google Docs](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) with pw: fstq \
If you don't have GPU, please use `--no_cuda` to run on cpu mode.

```
# crop and align the original v1.1.0 
python .\crop_align.py --data ..\data\GestaltMatcherDB\v1.1.0\gmdb_images --save_dir .\data\GestaltMatcherDB\v1.1.0\gmdb_align

# crop and align the test image and obtained the aligned image in .\demo_images\cdls_demo_alinged.jpg
python .\crop_align.py --data .\demo_images\cdls_demo.png --save_dir .\demo_images\
```

### Train models
The training of GestaltMatcher-Arc needs to be run twice:
* a) for the resnet-50 mix model, and
* b) for the resnet-100 model.
For these also require the pretrained ArcFace models from insightface: `glint360k_r50.onnx` and `glint360k_r100.onnx` to 
be in the directory `./saved_models`. \
These models can be downloaded here: https://github.com/deepinsight/insightface/tree/master/model_zoo 

The pretrained models by default are stored in a directory set by `--weight_dir` (default:`./saved_models/`). Further, 
using the arguments `--model_a_path`, `--model_b_path` and `--model_c_path`, the paths within this directory need to be
specified (default: uses all supplied model names). \
When setting any of those to 'None' they will not be included in the ensemble.

To reproduce our Gestalt Matcher model listed in the table by training from scratch, use:
```
python train_gm_arc.py --paper_model a --epochs 50 --session 1 --dataset gmdb --in_channels 3 --img_size 112 --use_tensorboard --local --data_dir ../data --dataset_version v1.1.0 
python train_gm_arc.py --paper_model b --epochs 50 --session 2 --dataset gmdb --in_channels 3 --img_size 112 --use_tensorboard --local --data_dir ../data --dataset_version v1.1.0 
```

You may choose whatever seed and session you find useful.
`--seed 11` was used to obtain these results, others have not been tested.

Using the argument `--use_tensorboard` allows you to track your models training and validation curves over time.

Training a model without GPU has not been tested.

### Encode photos
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
```
# encode the whole GMDB dataset to obtain the gallery encodings
python predict.py 
  --model_a_path s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth 
  --model_b_path s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth 
  --save_as_pickle 
  --data ../data/GestaltMatcherDB/v1.1.0/gmdb_align/ 
  --save_dir ./data/gallery_encodings/ 
  --output_name GMDB_gallery_encodings_v1.1.0.pkl
  
 # encode the image you want to test
 # the output will be in ./data/demo_test/test_encodings_v1.1.0.pkl
 python predict.py 
  --model_a_path s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth 
  --model_b_path s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth 
  --save_as_pickle 
  --data demo_images/cdls_demo_aligned.jpg 
  --save_dir ./data/demo_test/ 
  --output_name test_encodings_v1.1.0.pkl
```

There are 12 encodings per image because there are three models, and test-time augmentation including flip and
color/grey. Please find more detail in our paper [Hustinx et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Hustinx_Improving_Deep_Facial_Phenotyping_for_Ultra-Rare_Disorder_Verification_Using_Model_WACV_2023_paper.pdf). 

The structure of the file is shown below.

| Field                       | Value                                                                    |
| -------------------- |--------------------------------------------------------------------------|
| `img_name`          | `cdls_demo_aligned.jpg`                                                  |
| `model`             | `m0, m1, or m2`                                                          |
| `flip`              | `0 or 1`                                                                 |
| `gray`              | `0 or 1`                                                                 |
| `class_conf`        | `[19.153106689453125, -1.1602606773376465, ...]` (truncated for brevity) |
| `representations` | `[1.054652452468872, 0.4113105237483978, ...]` (truncated for brevity)   |

### Evaluation
#### Evaluate on the whole dataset
The following result is evaluating the whole v1.1.0 GMDB dataset.
```
python .\evaluate_ensemble.py

===========================================================
---------   test: Frequent, gallery: Frequent    ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-frequent|8794    |882   |42.37 |65.48 |71.07 |83.68 |
---------       test: Rare, gallery: Rare        ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |922.6   |386.4 |32.61 |47.22 |54.61 |68.57 |
--------- test: Frequent, gallery: Frequent+Rare ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-frequent|9716.6  |882   |42.10 |63.79 |70.49 |81.51 |
---------   test: Rare, gallery: Frequent+Rare   ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |9716.6  |386.4 |19.74 |31.26 |37.59 |48.58 |
===========================================================
```

#### Evaluate using Mulit-Image GestaltMatcher
Our recent work has explored option for combining multiple images, either multiple (test) images belonging to the same 
patient, or multiple gallery images of patients with the same disorder. We have released an evaluation script that 
contains the most important configurations and combinations of our experiments such as to reproduce the results in our 
preprint [*link here*].

Following the earlier data preparation steps, you can simply use it out the box with the following commands:\
`python .\evaluate_ensemble_multi_image.py`, for the entire test set and\
`python .\evaluate_ensemble_multi_image.py --multi_only`, using a subset containing only patients with multiple images.

A comparison of the most important results is shown below. 
```
                Multi-Image GestaltMatcher 
======================================================
-------   test: Frequent, gallery: Frequent    -------
|Test set       |Gallery |Test  |Top-1 |Top-5 |Top-10|
|GMDB-frequent  |8794    |882   |61.42 |80.03 |84.57 |
|GMDB-frequent* |8794    |330   |71.56 |88.07 |92.66 |
-------       test: Rare, gallery: Rare        -------
|Test set       |Gallery |Test  |Top-1 |Top-5 |Top-10|
|GMDB-rare      |922.6   |386.4 |35.75 |50.62 |58.04 |
|GMDB-rare*     |922.6   |126.8 |39.52 |55.42 |62.69 |
======================================================

<test set>* indicates the multi-only subset, containing 
only patients with more than 1 test image.
```

There are multiple parameters that can be set for such that users don't need to adjust the code too much:

| Argument      | Description                                                                                                                              |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `split`       | Test and gallery splits to use. Options: `all`, `ff`, `rr`, `fa`, `ra`. (default=`all`)                                                  |
| `version`     | GMDB version to evaluate. (default=`v1.1.0`)                                                                                             |
| `multi_only`  | Flag to set if you only want to evaluate on multiple images.                                                                             |
| `encodings_path` | Path to the file containing GM encodings of all images. Supported types: `csv`, `pkl`, and `p` (default=`./encodings/all_encodings.csv`) |
| `metadata_path`  | Path to the directory containing metadata-files. (default=`../data/GestaltMatcherDB/v1.1.0/gmdb_metadata`)|

#### Evaluate encodings with gallery encodings
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

```
python evaluate.py --metadata_dir ../data/GestaltMatcherDB/v1.1.0/gmdb_metadata
--gallery_input ./data/gallery_encodings/GMDB_gallery_encodings_v1.1.0.pkl
--output_dir demo_output --output_file demo_results.json
--case_input ./data/demo_test/test_encodings_v1.1.0.pkl --top_n all
```

## References
1. **GestaltMatcher**: Hsieh, T.-C. et al. (2022). GestaltMatcher facilitates rare disease matching using facial phenotype descriptors. Nature Genetics, 54(3), 349-357. [https://www.nature.com/articles/s41588-021-01010-x](https://www.nature.com/articles/s41588-021-01010-x)
2. **GestaltMatcher-Arc**: Hustinx, A. et al. (2023). Improving deep facial phenotyping for ultra-rare disorder verification using model ensembles. 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). doi:[10.1109/wacv56688.2023.00499](https://openaccess.thecvf.com/content/WACV2023/papers/Hustinx_Improving_Deep_Facial_Phenotyping_for_Ultra-Rare_Disorder_Verification_Using_Model_WACV_2023_paper.pdf)
3. **GestaltMatcher Database**: Lesmann, H. et al. (2024). GestaltMatcher Database - A global reference for facial phenotypic variability in rare human diseases. medRxiv. doi:[10.1101/2023.06.06.23290887](https://www.medrxiv.org/content/10.1101/2023.06.06.23290887v3)

## Contact
Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
