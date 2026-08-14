#!/usr/bin/env python3
"""
Download and assemble the UCI "Human Activity Recognition Using Smartphones"
dataset into the flat train.csv / test.csv shape expected by
human_activity_recognition.ipynb (561 sensor features + `subject` +
`Activity` columns).

Source: UCI Machine Learning Repository (public, no authentication required)
    https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

The UCI archive ships the dataset as separate whitespace-delimited files
(X_train.txt, y_train.txt, subject_train.txt, features.txt, ...) rather than
a single CSV. This script downloads that archive, reads the raw files, and
assembles them into data/train.csv and data/test.csv.

Usage:
    python download_data.py
    python download_data.py --force   # re-download even if data/ already exists
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

MANUAL_INSTRUCTIONS = """
Could not download the dataset automatically.

To get the data manually:

1. Go to the UCI Machine Learning Repository page for the "Human Activity
   Recognition Using Smartphones" dataset:
       https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Download the dataset zip file from that page.
3. Unzip it. Inside you should find a nested "UCI HAR Dataset.zip" — unzip
   that too. You should end up with a "UCI HAR Dataset" folder containing
   features.txt, activity_labels.txt, and train/ and test/ subfolders
   (each with X_*.txt, y_*.txt, and subject_*.txt files).
4. Re-run this script pointing at that folder:
       python download_data.py --source-dir "/path/to/UCI HAR Dataset"
   This will skip the download and just assemble data/train.csv and
   data/test.csv from the local files.

Alternatively, if you have a Kaggle account, the same dataset (already in
the flat train.csv / test.csv shape this notebook expects) is mirrored at:
       https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
Download train.csv and test.csv from there and place them directly in the
data/ directory.
"""


def download_zip(url: str, dest: Path, timeout: int = 60) -> None:
    """Stream-download a URL to a local file."""
    print(f"Downloading dataset from:\n  {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:6.1f} MB / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="")
        print()


def extract_dataset(outer_zip: Path, work_dir: Path) -> Path:
    """
    The UCI archive is a zip containing a *second* zip ("UCI HAR Dataset.zip").
    Extract both layers and return the path to the "UCI HAR Dataset" folder.
    """
    print("Extracting archive...")
    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(work_dir)

    inner_zip = work_dir / "UCI HAR Dataset.zip"
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(work_dir)

    dataset_dir = work_dir / "UCI HAR Dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Expected to find a 'UCI HAR Dataset' folder after extraction, "
            f"but it wasn't there. Contents of {work_dir}: {list(work_dir.iterdir())}"
        )
    return dataset_dir


def load_feature_names(dataset_dir: Path) -> list[str]:
    """
    Read features.txt and de-duplicate repeated names the same way pandas
    mangles duplicate CSV headers (name, name.1, name.2, ...). ~40 of the
    561 official feature names are reused across the X/Y/Z axes of the
    bandsEnergy() features, so de-duplication is required to get unique
    column names.
    """
    features = pd.read_csv(
        dataset_dir / "features.txt", sep=r"\s+", header=None, names=["idx", "name"]
    )
    names = features["name"].tolist()

    seen: dict[str, int] = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    return unique_names


def load_activity_map(dataset_dir: Path) -> dict[int, str]:
    labels = pd.read_csv(
        dataset_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"]
    )
    return dict(zip(labels["id"], labels["name"]))


def assemble_split(dataset_dir: Path, split: str, feature_names: list[str], activity_map: dict[int, str]) -> pd.DataFrame:
    """Build one flat DataFrame (X + subject + Activity) for 'train' or 'test'."""
    split_dir = dataset_dir / split

    X = pd.read_csv(split_dir / f"X_{split}.txt", sep=r"\s+", header=None, names=feature_names)
    y = pd.read_csv(split_dir / f"y_{split}.txt", sep=r"\s+", header=None, names=["activity_id"])
    subject = pd.read_csv(split_dir / f"subject_{split}.txt", sep=r"\s+", header=None, names=["subject"])

    df = X.copy()
    df["subject"] = subject["subject"]
    df["Activity"] = y["activity_id"].map(activity_map)
    return df


def assemble_from_local(dataset_dir: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = load_feature_names(dataset_dir)
    activity_map = load_activity_map(dataset_dir)

    for split in ("train", "test"):
        print(f"Assembling {split}.csv ...")
        df = assemble_split(dataset_dir, split, feature_names, activity_map)
        out_path = DATA_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download/rebuild even if data/ already exists"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to an already-extracted 'UCI HAR Dataset' folder (skips download)",
    )
    args = parser.parse_args()

    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    if train_csv.exists() and test_csv.exists() and not args.force:
        print(f"{train_csv} and {test_csv} already exist. Use --force to rebuild.")
        return 0

    if args.source_dir:
        dataset_dir = Path(args.source_dir).expanduser().resolve()
        if not dataset_dir.exists():
            print(f"--source-dir does not exist: {dataset_dir}", file=sys.stderr)
            return 1
        assemble_from_local(dataset_dir)
        print("\nDone. data/train.csv and data/test.csv are ready.")
        return 0

    try:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            outer_zip = work_dir / "har.zip"
            download_zip(DATASET_URL, outer_zip)
            dataset_dir = extract_dataset(outer_zip, work_dir)
            assemble_from_local(dataset_dir)
    except Exception as exc:  # noqa: BLE001 - want to catch network/parse errors alike
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        return 1

    print("\nDone. data/train.csv and data/test.csv are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
