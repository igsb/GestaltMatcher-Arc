import base64
import pickle
import secrets
from typing import Annotated
from lib.encode import *
from lib.evaluation import *
from pydantic import BaseModel
from lib.face_alignment import *
from contextlib import asynccontextmanager
from lib.utils_functions import readb64, encodeb64
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

with open('config.json', 'r') as config_file:
    print(1233445)
    config = json.load(config_file)

USERNAME = config.get('username')
PASSWORD = config.get('password')

def get_current_username(
        credentials: Annotated[HTTPBasicCredentials, Depends(security)]
    ):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = USERNAME.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = PASSWORD.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


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
    with open(os.path.join("data", "image_gene_and_syndrome_metadata_10102023.p"), "rb") as f:
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
async def predict_endpoint(username: Annotated[str, Depends(get_current_username)], image: Img):
    img = readb64(image.img)

    start_time = time.time()
    timestamp = time.time()

    # Convert the timestamp to a datetime object
    datetime_obj = datetime.fromtimestamp(timestamp)

    # Format the datetime object as a readable string
    formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')

    print("Formatted Time:", formatted_time)
    try:
        aligned_img = face_align_crop(_cropper_model, img, _device)
    except Exception as e:
        return {"message": "Face alignment error."}
    align_time = time.time()
    try:
        encoding = encode(_models, 'cpu', aligned_img, False, False)
    except Exception as e:
        return {"message": "Encoding error."}
    encode_time = time.time()
    try:
        result = predict(encoding,
                         _gallery_df,
                         _images_synds_dict,
                         _images_genes_dict,
                         _genes_metadata_dict,
                         _synds_metadata_dict)
    except Exception as e:
        return {"message": "Evaluation error."}
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
