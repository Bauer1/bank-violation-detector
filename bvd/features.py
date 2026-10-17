"""Per-frame feature extraction: CLIP embeddings + zero-shot prompt scores,
YOLO people/objects layout, motion energy."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import (CFG, COCO_BAGS, COCO_BOOK, COCO_PERSON, COCO_PHONE,
                     PROMPTS, YOLO_CLASSES)
from .video import read_frames

N_TOP_PERSONS = 3
MOTION_GRID = 4
FEATURE_SCHEMA = 2


def _pad_square(img: np.ndarray) -> Image.Image:
    """CLIP центр-кропает кадр; для вертикального видео это теряет половину сцены,
    поэтому дополняем до квадрата."""
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return Image.fromarray(canvas)


def motion_features(frames: list[np.ndarray]) -> np.ndarray:
    g = MOTION_GRID
    out = np.zeros((len(frames), 1 + g * g), dtype=np.float32)
    prev = None
    for i, f in enumerate(frames):
        cur = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32) / 255.0
        if prev is not None:
            d = np.abs(cur - prev)
            out[i, 0] = d.mean()
            out[i, 1:] = d.reshape(g, 64 // g, g, 64 // g).mean(axis=(1, 3)).ravel()
        prev = cur
    if len(frames) > 1:
        out[0] = out[1]
    return out


def mirrored_motion_features(dets: list[np.ndarray]) -> np.ndarray:
    """Estimate whether a pair of detected people behaves like a reflection.

    Reflected detections tend to have similar boxes and vertical motion, while
    their horizontal positions are mirrored and horizontal motion is opposite.
    The score is deliberately conservative and is only an input feature.
    """
    tracks: list[dict[int, tuple[float, float, float, float]]] = []
    active: list[int] = []
    for frame_idx, d in enumerate(dets):
        people = d[d[:, 5] == COCO_PERSON]
        boxes = []
        for x1, y1, x2, y2, _conf, _cls in people:
            boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))

        candidates = []
        for track_idx in active:
            last_frame = max(tracks[track_idx])
            if last_frame != frame_idx - 1:
                continue
            px, py, pw, ph = tracks[track_idx][last_frame]
            for box_idx, (cx, cy, bw, bh) in enumerate(boxes):
                distance = float(np.hypot(cx - px, cy - py))
                # Include scale change in the association cost; refuse large jumps.
                scale_delta = abs(np.log((bw + 1e-3) / (pw + 1e-3))) + abs(np.log((bh + 1e-3) / (ph + 1e-3)))
                cost = distance + 0.04 * scale_delta
                if distance <= 0.22:
                    candidates.append((cost, track_idx, box_idx))
        candidates.sort()
        used_tracks, used_boxes = set(), set()
        for _cost, track_idx, box_idx in candidates:
            if track_idx in used_tracks or box_idx in used_boxes:
                continue
            tracks[track_idx][frame_idx] = boxes[box_idx]
            used_tracks.add(track_idx)
            used_boxes.add(box_idx)
        for box_idx, box in enumerate(boxes):
            if box_idx not in used_boxes:
                tracks.append({frame_idx: box})
                active.append(len(tracks) - 1)

    scores = np.zeros(len(dets), dtype=np.float32)
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            shared = sorted(set(tracks[a]).intersection(tracks[b]))
            if len(shared) < 4:
                continue
            # Evaluate a short local trajectory around each frame, not the full clip.
            for center in shared:
                idx = [i for i in shared if abs(i - center) <= 2]
                if len(idx) < 4:
                    continue
                p, q = np.asarray([tracks[a][i] for i in idx]), np.asarray([tracks[b][i] for i in idx])
                deltas = np.diff(idx)
                consecutive = deltas == 1
                if consecutive.sum() < 3:
                    continue
                dp, dq = np.diff(p, axis=0)[consecutive], np.diff(q, axis=0)[consecutive]

                # Similar stature/vertical placement and a stable mirrored axis.
                h = max(float(np.mean((p[:, 3] + q[:, 3]) / 2)), 1e-3)
                w = max(float(np.mean((p[:, 2] + q[:, 2]) / 2)), 1e-3)
                geometry = np.exp(-float(np.std(p[:, 0] + q[:, 0])) / (0.5 * w + 1e-3))
                geometry *= np.exp(-float(np.mean(np.abs(p[:, 1] - q[:, 1]))) / (0.45 * h + 1e-3))
                geometry *= np.exp(-float(np.mean(np.abs(p[:, 2] - q[:, 2]))) / (0.6 * w + 1e-3))
                geometry *= np.exp(-float(np.mean(np.abs(p[:, 3] - q[:, 3]))) / (0.6 * h + 1e-3))

                horizontal_travel = float(np.abs(dp[:, 0]).sum() + np.abs(dq[:, 0]).sum())
                if horizontal_travel < 0.025:
                    continue
                x_corr = 0.5  # no useful horizontal variation
                if np.std(dp[:, 0]) > 1e-3 and np.std(dq[:, 0]) > 1e-3:
                    x_corr = max(0.0, float(np.corrcoef(dp[:, 0], -dq[:, 0])[0, 1]))
                y_corr = 0.5  # vertical motion is often too small to correlate
                if np.std(dp[:, 1]) > 1e-3 and np.std(dq[:, 1]) > 1e-3:
                    y_corr = max(0.0, float(np.corrcoef(dp[:, 1], dq[:, 1])[0, 1]))
                score = float(geometry * (0.8 * x_corr + 0.2 * y_corr))
                for i in idx:
                    scores[i] = max(scores[i], score)
    return scores


class FeatureExtractor:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Keep inference batches modest on laptop GPUs such as the RTX 2050 4 GB.
        self.clip_batch_size = 8 if self.device == "cuda" else 32
        self.yolo_batch_size = 4 if self.device == "cuda" else 16
        import open_clip
        from ultralytics import YOLO

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            CFG.clip_model, pretrained=CFG.clip_pretrained, device=self.device)
        self.clip.eval()
        tok = open_clip.get_tokenizer(CFG.clip_model)
        with torch.no_grad():
            t = self.clip.encode_text(tok(PROMPTS).to(self.device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.yolo = YOLO(CFG.yolo_weights)

    # ---------------- CLIP ----------------
    @torch.no_grad()
    def clip_features(self, frames, bs: int | None = None):
        bs = bs or self.clip_batch_size
        embs = []
        for i in range(0, len(frames), bs):
            batch = torch.stack([self.preprocess(_pad_square(f)) for f in frames[i:i + bs]]).to(self.device)
            with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                e = self.clip.encode_image(batch)
            e = e.float()
            embs.append(e / e.norm(dim=-1, keepdim=True))
        E = torch.cat(embs)
        sims = E @ self.text.T
        return E.cpu().numpy().astype(np.float32), sims.cpu().numpy().astype(np.float32)

    # ---------------- YOLO ----------------
    def detect(self, frames, bs: int | None = None):
        """Список детекций на кадр: array (N, 6) = x1,y1,x2,y2 (норм.), conf, cls."""
        bs = bs or self.yolo_batch_size
        dets = []
        for i in range(0, len(frames), bs):
            bgr = [np.ascontiguousarray(f[..., ::-1]) for f in frames[i:i + bs]]
            res = self.yolo.predict(bgr, classes=YOLO_CLASSES, conf=CFG.yolo_conf, imgsz=640,
                                    device=0 if self.device == "cuda" else "cpu",
                                    half=self.device == "cuda", verbose=False)
            for r in res:
                b = r.boxes
                if b is None or len(b) == 0:
                    dets.append(np.zeros((0, 6), np.float32))
                else:
                    dets.append(np.hstack([b.xyxyn.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                           b.cls.cpu().numpy()[:, None]]).astype(np.float32))
        return dets

    @staticmethod
    def layout_features(dets) -> np.ndarray:
        rows = []
        for d in dets:
            p = d[d[:, 5] == COCO_PERSON]
            bags = d[np.isin(d[:, 5], COCO_BAGS)]
            phone = d[d[:, 5] == COCO_PHONE]
            book = d[d[:, 5] == COCO_BOOK]
            f = [len(p), float((p[:, 4] > 0.5).sum()) if len(p) else 0.0]
            if len(p):
                w, h = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
                area = w * h
                f += [area.sum(), h.max(), (h / np.maximum(w, 1e-3)).max()]
                order = np.argsort(-area)[:N_TOP_PERSONS]
            else:
                f += [0.0, 0.0, 0.0]
                order = []
            for j in range(N_TOP_PERSONS):
                if j < len(order):
                    b = p[order[j]]
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    f += [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, bw, bh, b[4]]
                else:
                    f += [0.0] * 5
            for grp in (bags, phone, book):
                f += [len(grp), grp[:, 4].max() if len(grp) else 0.0]
            rows.append(f)
        return np.asarray(rows, dtype=np.float32)

    # ---------------- all ----------------
    def extract(self, times, frames, duration) -> dict:
        clip_emb, sims = self.clip_features(frames)
        dets = self.detect(frames)
        layout = self.layout_features(dets)
        mirrored_motion = mirrored_motion_features(dets)
        motion = motion_features(frames)
        low = np.hstack([sims * 10.0, layout, motion * 10.0, mirrored_motion[:, None]]).astype(np.float32)
        return {"times": np.asarray(times, np.float32), "clip": clip_emb, "low": low,
                "duration": np.float32(duration)}

    def extract_path(self, path: str | Path):
        times, frames, duration = read_frames(path)
        return self.extract(times, frames, duration), frames


def _cache_key(path: Path) -> str:
    st = path.stat()
    sig = f"v{FEATURE_SCHEMA}|{path.name}|{st.st_size}|{CFG.sample_fps}|{CFG.clip_model}|{CFG.yolo_weights}|{len(PROMPTS)}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def cached_features(extractor_factory, path: Path) -> dict:
    CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    fp = CFG.cache_dir / f"{path.stem}_{_cache_key(path)}.npz"
    if fp.exists():
        z = np.load(fp)
        return {k: z[k] for k in z.files}
    feat, _ = extractor_factory().extract_path(path)
    np.savez_compressed(fp, **feat)
    return feat
