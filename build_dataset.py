"""Turn CHB-MIT EDF files into a TorchEEG MNERawDataset."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from src.parse_summary import parse_summary

COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-0",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

WINDOW_SEC = 4.0
SFREQ = 256.0
BUFFER_SEC = 30.0
NONSEIZURE_RATIO = 3.0


def _pick_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    lookup = {}
    for name in raw.ch_names:
        key = name.upper().replace("EEG", "").replace(" ", "")
        lookup.setdefault(key, name)
    picked = []
    wanted = [ch.replace(" ", "").upper() for ch in COMMON_CHANNELS]
    wanted = ["T8-P8" if ch == "T8-P8-0" else ch for ch in wanted]
    for key in wanted:
        if key in lookup and lookup[key] not in picked:
            picked.append(lookup[key])
        elif key == "T8-P8":
            for alt in ("T8-P8-0", "T8-P8"):
                if alt in lookup and lookup[alt] not in picked:
                    picked.append(lookup[alt])
                    break
    if len(picked) < 16:
        raise ValueError(f"Too few shared channels in {raw.filenames}: {raw.ch_names}")
    return raw.pick(picked)


def _preprocess(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.load_data()
    raw.notch_filter(freqs=60.0, verbose=False)
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    return _pick_channels(raw)


def _crop_copy(raw: mne.io.BaseRaw, tmin: float, tmax: float) -> mne.io.BaseRaw | None:
    tmax = min(tmax, raw.times[-1])
    tmin = max(tmin, 0.0)
    if tmax - tmin < WINDOW_SEC:
        return None
    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    return cropped


def _nonseizure_intervals(duration: float, seizures, buffer_sec: float) -> list[tuple[float, float]]:
    blocked = []
    for seizure in seizures:
        blocked.append((max(0.0, seizure.start_sec - buffer_sec), min(duration, seizure.end_sec + buffer_sec)))
    blocked.sort()
    merged = []
    for start, end in blocked:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    intervals = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= WINDOW_SEC:
            intervals.append((cursor, start))
        cursor = end
    if duration - cursor >= WINDOW_SEC:
        intervals.append((cursor, duration))
    return intervals


def build_raw_lists(raw_dir: Path, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(42)
    summary_path = next(raw_dir.glob("*-summary.txt"))
    records = parse_summary(summary_path)
    raw_list = []
    info_list = []
    trial_id = 0

    for record in records:
        edf_path = raw_dir / record.file_name
        if not edf_path.exists():
            continue
        raw = _preprocess(mne.io.read_raw_edf(edf_path, preload=False, verbose=False))
        duration = float(raw.times[-1])
        split = _file_split(record.file_name)

        for seizure in record.seizures:
            cropped = _crop_copy(raw, seizure.start_sec, seizure.end_sec)
            if cropped is None:
                continue
            raw_list.append(cropped)
            info_list.append(
                {
                    "label": 1,
                    "subject_id": raw_dir.name,
                    "file_name": record.file_name,
                    "trial_id": trial_id,
                    "split": split,
                }
            )
            trial_id += 1

        seizure_duration = sum(s.end_sec - s.start_sec for s in record.seizures)
        target_nonseizure = max(WINDOW_SEC * 8, seizure_duration * NONSEIZURE_RATIO if seizure_duration else 120.0)
        intervals = _nonseizure_intervals(duration, record.seizures, BUFFER_SEC)
        collected = 0.0
        rng.shuffle(intervals)
        for start, end in intervals:
            remain = target_nonseizure - collected
            if remain <= 0:
                break
            take = min(end - start, max(remain, WINDOW_SEC))
            offset = 0.0 if end - start <= take else float(rng.uniform(0, end - start - take))
            cropped = _crop_copy(raw, start + offset, start + offset + take)
            if cropped is None:
                continue
            raw_list.append(cropped)
            info_list.append(
                {
                    "label": 0,
                    "subject_id": raw_dir.name,
                    "file_name": record.file_name,
                    "trial_id": trial_id,
                    "split": split,
                }
            )
            trial_id += 1
            collected += take

    if not raw_list:
        raise FileNotFoundError(f"No EDF files found under {raw_dir}")
    return raw_list, info_list


def _file_split(file_name: str) -> str:
    # File-wise split to reduce leakage between adjacent windows.
    test_files = {"chb01_21.edf", "chb01_26.edf", "chb01_05.edf"}
    val_files = {"chb01_18.edf", "chb01_02.edf"}
    if file_name in test_files:
        return "test"
    if file_name in val_files:
        return "val"
    return "train"


def make_torcheeg_dataset(raw_dir: Path, io_path: Path):
    from torcheeg.datasets.module.mne_raw_dataset import MNERawDataset
    from torcheeg.transforms.any.compose import Compose
    from torcheeg.transforms.label.select import Select
    from torcheeg.transforms.numpy.normalize import MeanStdNormalize
    from torcheeg.transforms.numpy.to import To2d
    from torcheeg.transforms.torch.to import ToTensor

    raw_list, info_list = build_raw_lists(raw_dir)
    chunk_size = int(WINDOW_SEC * SFREQ)
    dataset = MNERawDataset(
        raw_list=raw_list,
        info_list=info_list,
        chunk_size=chunk_size,
        overlap=int(chunk_size * 0.5),
        online_transform=Compose(
            [
                MeanStdNormalize(),
                To2d(),
                ToTensor(),
            ]
        ),
        label_transform=Select("label"),
        io_path=str(io_path),
        io_mode="pickle",
        io_size=512 * 1024 * 1024,
        num_worker=0,
        verbose=True,
    )
    return dataset, info_list, chunk_size
