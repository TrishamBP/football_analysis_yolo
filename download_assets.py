# /// script
# requires-python = ">=3.12"
# dependencies = ["gdown"]
# ///
"""Download the model weights and demo video needed by main.py.

Usage: uv run download_assets.py
"""
import os

import gdown

ASSETS = {
    # YOLOv8x player detection model (ball/goalkeeper/player/referee) from github.com/roboflow/sports
    "models/best.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    # 30s DFL Bundesliga broadcast clip (08fd33_0.mp4)
    "data/test.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
}


def main():
    for path, file_id in ASSETS.items():
        if os.path.exists(path):
            print(f"{path} already exists, skipping")
            continue

        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(id=file_id, output=path)

        if not os.path.exists(path):
            raise SystemExit(f"Failed to download {path}")

    print("Assets ready. Run the pipeline with: uv run main.py")


if __name__ == "__main__":
    main()
