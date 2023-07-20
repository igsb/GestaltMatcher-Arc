import base64
import pickle
from lib.encode import *
from lib.evaluation import *
from fastapi import FastAPI
from pydantic import BaseModel
from lib.face_alignment import *
from contextlib import asynccontextmanager
from lib.utils_functions import readb64, encodeb64

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models
    global _device
    global _cropper_model
    global _gallery_df
    global _images_synds_dict
    global _images_genes_dict
    global _genes_metadata_dict
    global _synds_metadata_dict
    _models = get_models()
    _cropper_model, _device = load_cropper_model()
    # Load synd dict
    with open(os.path.join("data", "image_gene_and_syndrome_metadata_v1.0.3.p"), "rb") as f:
        data = pickle.load(f)
    _images_synds_dict = data["disorder_level_metadata"]
    _images_genes_dict = data["gene_level_metadata"]
    _genes_metadata_dict = data["gene_metadata"]
    _synds_metadata_dict = data["disorder_metadata"]
    _gallery_df = get_gallery_encodings_set(_images_synds_dict)

    yield


app = FastAPI(lifespan=lifespan)


class Img(BaseModel):
    img: str

@app.post("/predict")
async def predict_endpoint(image: Img):
    img = readb64(image.img)
    start_time = time.time()

    aligned_img = face_align_crop(_cropper_model, img, _device)
    align_time = time.time()

    encoding = encode(_models, 'cpu', aligned_img)
    encode_time = time.time()

    result = predict(encoding,
                     _gallery_df,
                     _images_synds_dict,
                     _images_genes_dict,
                     _genes_metadata_dict,
                     _synds_metadata_dict)
    finished_time = time.time()

    print('Crop: {:.2f}s'.format(align_time-start_time))
    print('Encode: {:.2f}s'.format(encode_time-align_time))
    print('Predict: {:.2f}s'.format(finished_time-encode_time))
    print('Total: {:.2f}s'.format(finished_time-start_time))
    return result

@app.post("/encode")
async def encode_endpoint(image: Img):
    img = readb64(image.img)
    aligned_img = face_align_crop(_cropper_model, img, _device)
    return {"encodings": encode(_models, 'cpu', aligned_img).to_dict()}


@app.post("/crop")
async def crop_endpoint(image: Img):
    #print(image)
    img = readb64(image.img)
    aligned_img = face_align_crop(_cropper_model, img, _device)
    img_en = cv2.imencode(".png", aligned_img)
    #aligned_img_bytes = aligned_img.tobytes()
    return {"crop": base64.b64encode(img_en[1])}


@app.get("/status")
async def status_endpoint():
    return {"status": "running"}

#if __name__ == "__main__":
#    global _models
#    # _models = []
#    #_models = get_models()
#    print(len(_models))
#    uvicorn.run("main:app", port=5000, log_level="info")