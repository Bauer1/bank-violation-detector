"""Video decoding / frame sampling."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import CFG


def read_frames(path: str | Path, fps: float = CFG.sample_fps, max_side: int = CFG.max_side):
    """Returns (times[s], frames[RGB uint8], duration[s])."""
    path = str(path)
    if not path.isascii():  # OpenCV на Windows не открывает пути с кириллицей
        tmp = Path(tempfile.gettempdir()) / f"bvd_{abs(hash(path))}{Path(path).suffix}"
        shutil.copyfile(path, tmp)
        try:
            return read_frames(tmp, fps, max_side)
        finally:
            tmp.unlink(missing_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # учитываем поворот .mov с телефона
    except Exception:
        pass
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0

    times, frames = [], []
    next_t, idx = 0.0, 0
    while True:
        if not cap.grab():
            break
        t = idx / src_fps
        if t + 1e-6 >= next_t:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                s = max_side / max(h, w)
                if s < 1:
                    fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                times.append(t)
            next_t += 1.0 / fps
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f"В видео нет кадров: {path}")
    return np.asarray(times, dtype=np.float32), frames, idx / src_fps
