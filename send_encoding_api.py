import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze collaborator-provided facial encodings via GestaltMatcher REST API."
    )
    parser.add_argument(
        "--case_input",
        required=True,
        help="Path to the input encoding .pkl file."
    )
    parser.add_argument(
        "--output_dir",
        default="embedding_output",
        help="Directory for saving one JSON result per case."
    )
    parser.add_argument(
        "--url",
        default="localhost",
        help="URL or hostname of the API."
    )
    parser.add_argument(
        "--port",
        default=5000,
        type=int,
        help="Port of the API."
    )
    parser.add_argument(
        "--username",
        required=True,
        help="API username."
    )
    parser.add_argument(
        "--password",
        required=True,
        help="API password."
    )
    return parser.parse_args()


def analyze_encoding(case_df, case_id, output_dir, api_endpoint, auth):
    """
    Send the embeddings for one case to the API and save its prediction JSON.
    """
    # Convert DataFrame values, including representation vectors, into JSON-safe Python lists.
    case_records = json.loads(case_df.to_json(orient="records"))
    params = {"encodings": case_records}

    response = requests.post(
        url=api_endpoint,
        json=params,
        auth=auth,
        timeout=300
    )

    try:
        data = response.json()
    except ValueError:
        print(f"Error for '{case_id}': API returned non-JSON output.")
        print(response.text)
        return False

    if response.status_code != 200:
        print(f"Error for '{case_id}' ({response.status_code}): {data}")
        return False

    output_data = {"case_id": case_id}
    output_data.update(data)

    output_filename = os.path.join(output_dir, f"{case_id}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    return True


def main():
    args = parse_args()

    required_columns = {
        "img_name", "model", "flip", "gray",
        "class_conf", "representations"
    }

    encoding_df = pd.read_pickle(args.case_input)

    missing_columns = required_columns - set(encoding_df.columns)
    if missing_columns:
        raise ValueError(
            f"Input encoding file is missing columns: {sorted(missing_columns)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    predict_url = f"http://{args.url}:{args.port}/predict_encoding"
    auth = requests.auth.HTTPBasicAuth(args.username, args.password)

    grouped_cases = list(encoding_df.groupby("img_name", sort=False))
    print(f"Start processing {len(grouped_cases)} encoded cases.")
    print(f"API endpoint: {predict_url}")

    start_time = time.time()
    successes = 0

    for count, (img_name, case_df) in enumerate(grouped_cases, start=1):
        case_id = Path(str(img_name)).stem

        ok = analyze_encoding(
            case_df=case_df.reset_index(drop=True),
            case_id=case_id,
            output_dir=args.output_dir,
            api_endpoint=predict_url,
            auth=auth
        )

        if ok:
            successes += 1
            print(f"Finished ({count}/{len(grouped_cases)}): {case_id}")
        else:
            print(f"Failed ({count}/{len(grouped_cases)}): {case_id}")

    elapsed = time.time() - start_time
    print(f"Successfully processed: {successes}/{len(grouped_cases)}")
    print(f"Total running time: {elapsed:.2f}s")
    print(f"Output JSON files are saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
