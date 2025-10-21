FROM python:3.10

WORKDIR /app

COPY requirements_docker.txt .

RUN pip install -r requirements_docker.txt
COPY saved_models/Resnet50_Final.pth ./saved_models/Resnet50_Final.pth
COPY saved_models/glint360k_r100.onnx ./saved_models/glint360k_r100.onnx
COPY saved_models/s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth ./saved_models/s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth
COPY saved_models/s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth ./saved_models/s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth
COPY data/gallery_encodings/GMDB_gallery_encodings_12062025_v1.1.0_service.pkl ./data/gallery_encodings/GMDB_gallery_encodings_12062025_v1.1.0_service.pkl
COPY main.py ./main.py
COPY data/image_gene_and_syndrome_metadata_pp4_12062025_max.p ./data/image_gene_and_syndrome_metadata_pp4_12062025_max.p
COPY data/transformation_probabilities_07052025.csv ./data/transformation_probabilities_07052025.csv
COPY config.json ./config.json
COPY lib ./lib

CMD [ "uvicorn",  "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]