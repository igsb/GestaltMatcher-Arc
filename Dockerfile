FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY saved_models/Resnet50_Final.pth ./saved_models/Resnet50_Final.pth
COPY saved_models/glint360k_r100.onnx ./saved_models/glint360k_r100.onnx
COPY saved_models/s107_glint360k_r50_512d_gmdb__v1.0.7_bs64_size112_channels3_last_model.pth ./saved_models/s107_glint360k_r50_512d_gmdb__v1.0.7_bs64_size112_channels3_last_model.pth
COPY saved_models/s107_glint360k_r100_512d_gmdb__v1.0.7_bs128_size112_channels3_last_model.pth ./saved_models/s107_glint360k_r100_512d_gmdb__v1.0.7_bs128_size112_channels3_last_model.pth
COPY data/gallery_encodings/GMDB_gallery_encodings_10102023_v1.0.7.pkl ./data/gallery_encodings/GMDB_gallery_encodings_10102023_v1.0.7.pkl
COPY main.py ./main.py
COPY data/image_gene_and_syndrome_metadata_10102023.p ./data/image_gene_and_syndrome_metadata_10102023.p
COPY config.json ./config.json
COPY lib ./lib

CMD [ "uvicorn",  "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
