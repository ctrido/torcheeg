"""Parse CHB-MIT summary.txt seizure annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Seizure:
    start_sec: float
    end_sec: float


@dataclass
class Record:
    file_name: str
    seizures: list[Seizure]


def parse_summary(path: Path) -> list[Record]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n(?=File Name:)", text)
    records: list[Record] = []
    for block in blocks:
        match = re.search(r"File Name:\s*(\S+)", block)
        if not match:
            continue
        file_name = match.group(1).strip()
        seizures: list[Seizure] = []
        starts = [float(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+(?:\.\d+)?)\s*seconds", block)]
        ends = [float(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+(?:\.\d+)?)\s*seconds", block)]
        for start, end in zip(starts, ends):
            seizures.append(Seizure(start_sec=start, end_sec=end))
        records.append(Record(file_name=file_name, seizures=seizures))
    return records
