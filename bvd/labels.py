"""Parsing of the organizers' Excel labels and matching to video files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CFG, VIDEO_TYPE

_CYR2LAT = str.maketrans({"Х": "X", "х": "X", "Х": "X"})


def normalize_id(name: str) -> str:
    """'Х-3' (кириллица), 'x_3', 'X-03.mov' -> 'X-3'."""
    s = Path(str(name)).stem.strip().translate(_CYR2LAT).upper()
    m = re.search(r"X\s*[-_ ]?\s*0*(\d+)", s)
    return f"X-{m.group(1)}" if m else s


def parse_timestamp(val) -> float | None:
    """'22 секунда' -> 22, '1минута 8 секунда' -> 68, NaN -> None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower()
    mins = re.search(r"(\d+)\s*мин", s)
    secs = re.search(r"(\d+)\s*сек", s)
    if not mins and not secs:
        nums = re.findall(r"\d+(?:[.,]\d+)?", s)
        return float(nums[0].replace(",", ".")) if nums else None
    return (int(mins.group(1)) * 60 if mins else 0) + (int(secs.group(1)) if secs else 0)


@dataclass
class VideoLabel:
    video_id: str
    violation: bool
    ts: float | None
    vtype: str | None


def load_labels(path: Path = CFG.labels_path) -> dict[str, VideoLabel]:
    df = pd.read_excel(path)
    df.columns = ["name", "ts", "label"][: len(df.columns)]
    out = {}
    for _, r in df.iterrows():
        vid = normalize_id(r["name"])
        viol = "без" not in str(r["label"]).lower()
        ts = parse_timestamp(r["ts"])
        out[vid] = VideoLabel(vid, viol, ts if viol else None, VIDEO_TYPE.get(vid) if viol else None)
    return out


def find_videos(video_dir: Path) -> dict[str, Path]:
    vids = {}
    for p in sorted(Path(video_dir).rglob("*")):
        if p.suffix.lower() in CFG.video_exts:
            vids[normalize_id(p.name)] = p
    return vids
