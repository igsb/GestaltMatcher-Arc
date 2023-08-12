FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY saved_models ./saved_models
COPY data/gallery_encodings/GMDB_gallery_encodings_10082023.pkl ./data/gallery_encodings/GMDB_gallery_encodings_10082023.pkl
COPY main.py ./main.py
COPY data/image_gene_and_syndrome_metadata_10082023.p ./data/image_gene_and_syndrome_metadata_10082023.p
COPY lib ./lib

CMD [ "uvicorn",  "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
