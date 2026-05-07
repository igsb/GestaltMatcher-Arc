import argparse
import os
from pathlib import Path

import pandas as pd
import requests


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(input_dir: Path):
    image_paths = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)
    return sorted(image_paths)


def encode_one_image(api_url: str, image_path: Path, timeout: int = 120) -> pd.DataFrame:
    endpoint = api_url.rstrip("/") + "/encode_file"

    with open(image_path, "rb") as f:
        files = {
            "file": (
                image_path.name,
                f,
                "application/octet-stream",
            )
        }
        response = requests.post(endpoint, files=files, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to encode {image_path.name}. "
            f"HTTP {response.status_code}: {response.text}"
        )

    payload = response.json()

    if "encodings" not in payload:
        raise RuntimeError(
            f"No 'encodings' field returned for {image_path.name}. "
            f"Response: {payload}"
        )

    df = pd.DataFrame(payload["encodings"])

    # main.py/lib.encode uses img_name='input', so we overwrite it with the real filename.
    df["img_name"] = image_path.name

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Send a folder of images to GestaltMatcher-Arc API and collect embeddings."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing images to encode.",
    )
    parser.add_argument(
        "--api_url",
        default="http://localhost:5000",
        help="Base URL of the GestaltMatcher-Arc API.",
    )
    parser.add_argument(
        "--output",
        default="case_encodings_v1.1.0.pkl",
        help="Output file. Use .pkl or .csv.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per image.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = collect_images(input_dir)

    if len(image_paths) == 0:
        raise RuntimeError(f"No supported image files found in: {input_dir}")

    print(f"Found {len(image_paths)} image(s).")

    all_dfs = []
    failed = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Encoding {image_path.name}")

        try:
            df = encode_one_image(
                api_url=args.api_url,
                image_path=image_path,
                timeout=args.timeout,
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"FAILED: {image_path.name}: {e}")
            failed.append(
                {
                    "img_name": image_path.name,
                    "error": str(e),
                }
            )

    if len(all_dfs) == 0:
        raise RuntimeError("No images were successfully encoded.")

    result_df = pd.concat(all_dfs, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".pkl":
        result_df.to_pickle(output_path)
    elif output_path.suffix.lower() == ".csv":
        result_df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output must end with .pkl or .csv")

    print(f"Saved encodings to: {output_path}")
    print(f"Successfully encoded: {len(all_dfs)} image(s)")
    print(f"Failed: {len(failed)} image(s)")

    if failed:
        failed_path = output_path.with_suffix(".failed.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"Saved failed image list to: {failed_path}")


if __name__ == "__main__":
    main()