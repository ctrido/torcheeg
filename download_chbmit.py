"""Download a CHB-MIT subject subset from PhysioNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"

DEFAULT_FILES = [
    "chb01-summary.txt",
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_05.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf",
]


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "torcheeg-chbmit-research"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)


def download_subject(subject: str, out_dir: Path, files: list[str] | None = None) -> Path:
    subject_dir = out_dir / subject
    files = files or DEFAULT_FILES
    for name in files:
        url = f"{PHYSIONET_BASE}/{subject}/{name}"
        download_file(url, subject_dir / name)
    return subject_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CHB-MIT files from PhysioNet")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--subject", default="chb01")
    args = parser.parse_args()
    path = download_subject(args.subject, args.out_dir)
    print(f"Downloaded files to {path.resolve()}")


if __name__ == "__main__":
    main()
