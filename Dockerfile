FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY saved_models ./saved_models
COPY __init__.py ./__init__.py
COPY data/gallery_encodings/GMDB_encodings_v1.0.3.pkl ./data/gallery_encodings/GMDB_encodings_v1.0.3.pkl
COPY main.py ./main.py
COPY data/image_gene_and_syndrome_metadata_v1.0.3.p ./data/image_gene_and_syndrome_metadata_v1.0.3.p
COPY lib ./lib

CMD [ "uvicorn",  "main:app", "--host", "0.0.0.0", "--port", "5000" ]
