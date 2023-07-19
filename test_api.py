import pandas as pd
import requests
import base64
import cv2
import os
import json
import time
import numpy as np

url_predix = "20.93.208.30"
url_prefix = "127.0.0.1"
# api-endpoint
URL = "http://{}:5000/crop".format(url_prefix)
#URL = "http://0.0.0.0:4000/crop"
encode_URL = "http://{}:5000/encode".format(url_prefix)
predict_URL = "http://{}:5000/predict".format(url_prefix)
status_URL = "http://{}:5000/status".format(url_prefix)

'''
r = requests.get(url=status_URL)
print(r.json())
'''

original_image = os.path.join("demo_images", "test.png")
original_image = os.path.join("demo_images", "1.jpg")
original_image = os.path.join("demo_images", "197.png")
with open(original_image, "rb") as f:
    img_raw_original = f.read()

encode_image = base64.b64encode(img_raw_original)
encode_image_str = encode_image.decode("utf-8")

# defining a params dict for the parameters to be sent to the API
PARAMS = {"img": encode_image_str}


#print(PARAMS)
# sending get request and saving the response as response object
r = requests.post(url=URL, json=PARAMS)
print(r)
# extracting data in json format
data = r.json()
#print(data["crop"])

nparr = np.fromstring(base64.b64decode(data["crop"]), np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv2.imwrite(os.path.join("test_output", f"1_aligned.jpg"), img)

r = requests.post(url=encode_URL, json=PARAMS)
print(r)
# extracting data in json format
data = r.json()
pd.DataFrame(data['encodings']).to_csv(os.path.join("test_output", "1.csv"), sep=";")

'''
start_time = time.time()
r = requests.post(url=predict_URL, json=PARAMS)
print(r)
# extracting data in json format
data = r.json()
#print(data['results'][:10])
with open(os.path.join("test_output", "1.json"), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
output_finished_time = time.time()
print('Predict: {:.2f}s'.format(output_finished_time-start_time))
'''