import os
import json
import time
import base64
import requests
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze image via REST api')
    parser.add_argument('--case_input', default='demo_images\cdls_demo.png', dest='case_input',
                        help='Path to the directory containing the case encodings or the single file containing the gallery encodings.')
    parser.add_argument('--output_dir', default='demo_output', dest='output_dir',
                        help='Path to the directory for saving the results.')
    parser.add_argument('--url', default='localhost', dest='url',
                        help='URL to the api.')
    parser.add_argument('--port', default=5000, dest='port',
                        help='Port to the api.')
    return parser.parse_args()


def analyze_image(input_file, output_dir, api_endpoint):
    file_predix = Path(input_file).stem
    with open(input_file, "rb") as f:
        img_raw_original = f.read()

    encode_image = base64.b64encode(img_raw_original)
    encode_image_str = encode_image.decode("utf-8")

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {"img": encode_image_str}

    auth = requests.auth.HTTPBasicAuth('your_username', 'your_password')
    r = requests.post(url=api_endpoint, json=PARAMS, auth=auth)
    # extracting data in json format
    status = r.status_code
    data = r.json()
    if status != 200:
        print(data)
    else:
        output_data = {}
        output_data['case_id'] = file_predix
        for key, value in data.items():
            output_data[key] = value
        output_filename = os.path.join(output_dir, "{}.json".format(file_predix))
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()

    # api-endpoint
    predict_URL = "http://{}:{}/predict".format(args.url, args.port)

    start_time = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    single_file = False
    if os.path.isfile(args.case_input):
        single_file = True

    # single file
    if single_file:
        print("Start processing {} file.".format(1))
        analyze_image(args.case_input, args.output_dir, predict_URL)
        print("Finished (1/1): {}".format(args.case_input))
    else:
        input_files = os.listdir(args.case_input)
        print("Start processing {} file.".format(len(input_files)))
        count = 1
        for input_file in input_files:
            filename = os.path.join(args.case_input, input_file)
            analyze_image(filename, args.output_dir, predict_URL)
            print("Finished ({}/{}): {}".format(count, len(input_files), filename))
            count += 1

    output_finished_time = time.time()
    print('Total running time: {:.2f}s'.format(output_finished_time - start_time))
    print('Output json files are saved in: {}'.format(args.output_dir))

if __name__ == '__main__':
    main()