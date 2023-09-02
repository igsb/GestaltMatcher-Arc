# GestaltMatcher-Arc: Service
This repository contains all the code for our GestaltMatcher service.
This repo also contains snippets of code from insightface (https://github.com/deepinsight/insightface); both from their 
alignment process and their RetinaFace detector.

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
2. s1_glint360k_r50_512d_gmdb__v1.0.3_bs64_size112_channels3_last_model.pth (model 1 for the encoding)
3. s2_glint360k_r100_512d_gmdb__v1.0.3_bs128_size112_channels3_last_model.pth (model 2 for the encoding)
4. glint360k_r100.onnx (model 3 for the encoding)

**Metadata**

Save the following file in ./data/
1. image_gene_and_syndrome_metadata_v1.0.3.p (image metadata)

**Encodings**

Save the following file in ./data/gallery_encodings/
1. GMDB_gallery_encodings_v1.0.3.pkl (image encodings)

### Build and run docker image
Build docker image: `docker build -t gm-api .`

Run and listen the request in localhost:5000:`docker run -p 5000:5000 gm-api`

### Send request
You can send a single image or multiple images in a folder to the api via **send_image_api.py**.
```
python send_image_api.py --case_input demo_images/cdls_demo.png --otuput_dir output

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

## Contact
Tzung-Chien Hsieh

Email: thsieh@uni-bonn.de or la60312@gmail.com

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
