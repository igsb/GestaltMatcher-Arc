import base64
import pickle
import secrets
import httpx
import numpy as np
from typing import Annotated
from lib.encode import *
from lib.evaluation import *
from pydantic import BaseModel
from lib.face_alignment import *
from contextlib import asynccontextmanager
from lib.utils_functions import readb64, encodeb64
from datetime import datetime

from pydantic import BaseModel, HttpUrl
from fastapi import Depends, Request, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated

security = HTTPBasic()


# generate a function to read csv file. each row has image id and patient id. Make a dictionary and each entry is image id, and the value is patient id.




with open('config.json', 'r') as config_file:
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
    global _synds_probabilities_dict
    _models = get_models()
    _cropper_model, _device = load_cropper_model()
    # Load synd dict

    with open(os.path.join("data", "image_gene_and_syndrome_metadata_pp4_19112025_v112.p"), "rb") as f:
        data = pickle.load(f)
    _images_synds_dict = data["disorder_level_metadata"]
    _images_genes_dict = data["gene_level_metadata"]
    _genes_metadata_dict = data["gene_metadata"]
    _synds_metadata_dict = data["disorder_metadata"]
    _gallery_df = get_gallery_encodings_set(_images_synds_dict)

    df = pd.read_csv(os.path.join("data", "transformation_probabilities_07052025.csv"), sep=",")

    _synds_probabilities_dict = {
        row["syndrome"]: {
            "(Intercept)": row["(Intercept)"],
            "syn_scores": row["syn_scores"],
            "v_00": row["v_00"],
            "v_10": row["v_10"],
            "v_11": row["v_11"]
        }
        for _, row in df.iterrows()
    }

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
                             _synds_metadata_dict,
                             _synds_probabilities_dict)
    except Exception as e:
        return {"message": "Evaluation error.", "error": str(e)}
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


@app.post("/predict_file")
async def predict_file_endpoint(
    request: Request,  # <-- move this before defaulted params
    username: Annotated[str, Depends(get_current_username)],
    file: UploadFile = File(...),
):
    # 1) Content-Type proves transport
    ct = request.headers.get("content-type", "")
    print("Content-Type:", ct)  # expect: multipart/form-data; boundary=...

    # 2) Size/filename proves it's a real uploaded file
    contents = await file.read()
    print("Uploaded:", file.filename, "bytes:", len(contents))

    img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    start_time = time.time()
    timestamp = time.time()
    datetime_obj = datetime.fromtimestamp(timestamp)
    formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
    print("Formatted Time:", formatted_time)

    try:
        aligned_img = face_align_crop(_cropper_model, img_array, _device)
    except Exception:
        return {"message": "Face alignment error."}
    align_time = time.time()
    try:
        encoding = encode(_models, 'cpu', aligned_img, False, False)
    except Exception:
        return {"message": "Encoding error."}
    encode_time = time.time()
    try:
        result = predict(
            encoding,
            _gallery_df,
            _images_synds_dict,
            _images_genes_dict,
            _genes_metadata_dict,
            _synds_metadata_dict
        )
    except Exception:
        return {"message": "Evaluation error."}
    finished_time = time.time()

    print('Crop: {:.2f}s'.format(align_time-start_time))
    print('Encode: {:.2f}s'.format(encode_time-align_time))
    print('Predict: {:.2f}s'.format(finished_time-encode_time))
    print('Total: {:.2f}s'.format(finished_time-start_time))
    return result


@app.post("/encode_file")
async def encode_file_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    aligned_img = face_align_crop(_cropper_model, img_array, _device)
    return {"encodings": encode(_models, 'cpu', aligned_img).to_dict()}


@app.post("/crop_file")
async def crop_file_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    aligned_img = face_align_crop(_cropper_model, img_array, _device)
    img_en = cv2.imencode(".png", aligned_img)
    return {"crop": base64.b64encode(img_en[1])}

import re, html
from urllib.parse import urlparse, parse_qs

def normalize_drive_url(u: str) -> str:
    """Turn Google Drive viewer/share links into a direct-download URL."""
    p = urlparse(u)
    if p.netloc in ("drive.google.com", "docs.google.com"):
        # /file/d/<id>/view
        m = re.search(r"/file/d/([^/]+)", p.path)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
        # ?id=<id> variants
        qs = parse_qs(p.query)
        if "id" in qs and qs["id"]:
            return f"https://drive.google.com/uc?export=download&id={qs['id'][0]}"
    return u


class ImgURL(BaseModel):
    url: HttpUrl  # e.g., https://drive.google.com/uc?export=download&id=...

@app.post("/predict_url")
async def predict_url_endpoint(
    username: Annotated[str, Depends(get_current_username)],
    payload: ImgURL,
):
    url = normalize_drive_url(str(payload.url))

    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        r = await client.get(url, headers={"User-Agent": "GM-API/1.0"})
        # If Drive serves an interstitial HTML, try to follow the confirm link once
        if "text/html" in r.headers.get("content-type","") and "confirm=" in r.text:
            m = re.search(r'href="(/uc\?export=download[^"]*confirm=[^"&]+[^"]*)"', r.text)
            if m:
                confirm_url = "https://drive.google.com" + html.unescape(m.group(1))
                r = await client.get(confirm_url, headers={"User-Agent": "GM-API/1.0"})
        r.raise_for_status()
        content = r.content

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")

    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image from URL")

    # 3) Your existing pipeline (unchanged)
    start_time = time.time()
    try:
        aligned_img = face_align_crop(_cropper_model, img, _device)
        encoding = encode(_models, 'cpu', aligned_img, False, False)
        result = predict(
            encoding,
            _gallery_df,
            _images_synds_dict,
            _images_genes_dict,
            _genes_metadata_dict,
            _synds_metadata_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    return result



@app.get("/status")
async def status_endpoint():
    return {"status": "running"}

#if __name__ == "__main__":
#    global _models
#    # _models = []
#    #_models = get_models()
#    print(len(_models))
#    uvicorn.run("main:app", port=5000, log_level="info")
