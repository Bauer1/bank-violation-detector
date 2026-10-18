"""Inference service object used by both FastAPI and Streamlit."""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import CFG, COCO_BAGS, COCO_PERSON, COCO_PHONE, VIOLATION_EXPLANATIONS, VIOLATION_TYPES
from .features import FeatureExtractor
from .video import read_frames

# Нужен для unpickle модели
from .model import ViolationModel  # noqa: F401

REGULATORY_CONTEXT = {
    "title": "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей",
    "document": "Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231",
    "url": "https://adilet.zan.kz/rus/docs/V1900019680",
    "note": "Общий нормативный контекст. По видеокадру нельзя подтвердить соблюдение конкретного пункта или операции.",
}


def _draw(frame_rgb: np.ndarray, dets: np.ndarray, title: str, bad: bool) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2, conf, cls in dets:
        cls = int(cls)
        color, name = (0, 200, 0), "person"
        if cls in COCO_BAGS:
            color, name = (0, 140, 255), "bag"
        elif cls == COCO_PHONE:
            color, name = (0, 0, 255), "phone"
        elif cls != COCO_PERSON:
            color, name = (200, 200, 0), "object"
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 180) if bad else (0, 120, 0), -1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Predictor:
    def __init__(self, model_path: str | Path = CFG.model_path, device: str | None = None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите scripts/train.py")
        self.model = joblib.load(model_path)
        # Порог из обучения (0.657) слишком консервативен: пропуск нарушения недопустим.
        self.model.threshold = CFG.inference_threshold
        self.extractor = FeatureExtractor(device)

    @staticmethod
    def _description(t: float, vtype: str | None) -> str:
        name = VIOLATION_TYPES.get(vtype, "возможное нарушение")
        why = VIOLATION_EXPLANATIONS.get(vtype, "наблюдаемые действия отличаются от обычного порядка работы кассы.")
        return (f"Около {t:.1f} с на видео: {name.lower()}. Что видно: {why} "
                f"Нормативный контекст: {REGULATORY_CONTEXT['document']} "
                f"«{REGULATORY_CONTEXT['title']}». Требуется проверка человеком.")

    def predict(self, video_path: str | Path) -> dict:
        t0 = time.time()
        times, frames, duration = read_frames(video_path)
        feat = self.extractor.extract(times, frames, duration)
        r = self.model.predict(feat)

        i = r["peak_idx"]
        dets = self.extractor.detect([frames[i]])[0]
        title = (f"VIOLATION  t={r['peak_time']:.1f}s  p={r['score']:.2f}" if r["violation"]
                 else f"OK  max p={r['score']:.2f} at {r['peak_time']:.1f}s")
        key = _draw(frames[i], dets, title, r["violation"])

        # 1–3 кадра-доказательства: размеченный пиковый кадр + соседние (±0.5 с при 2 fps)
        evidence = [(float(times[i]), key)]
        if r["violation"]:
            for j in (i - 1, i + 1):
                if 0 <= j < len(frames):
                    evidence.append((float(times[j]), frames[j]))
            evidence.sort(key=lambda e: e[0])

        return {
            "verdict": "обнаружены признаки возможных нарушений" if r["violation"] else "нарушение не обнаружено",
            "violation": r["violation"],
            "probability": round(r["score"], 4),
            "threshold": round(float(self.model.threshold), 4),
            "timestamp_sec": round(r["peak_time"], 1) if r["violation"] else None,
            "peak_time_sec": round(r["peak_time"], 1),
            "violation_type": r["type"] if r["violation"] else None,
            "violation_type_ru": VIOLATION_TYPES.get(r["type"]) if r["violation"] and r["type"] else None,
            "description": self._description(r["peak_time"], r["type"]) if r["violation"] else None,
            "regulatory_context": REGULATORY_CONTEXT if r["violation"] else None,
            "review_note": "Результат — подсказка для проверки человеком, а не юридическое или дисциплинарное заключение.",
            "type_probabilities": {k: round(v, 4) for k, v in r["type_probs"].items()},
            "duration_sec": round(float(duration), 1),
            "timeline": [{"t": round(float(t), 2), "p": round(float(p), 4)} for t, p in zip(r["times"], r["probs"])],
            "keyframe_rgb": key,
            "evidence_rgb": evidence,
            "processing_sec": round(time.time() - t0, 2),
        }

    @staticmethod
    def encode_jpg(img_rgb: np.ndarray) -> str | None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode() if ok else None

    @staticmethod
    def to_json(res: dict) -> dict:
        out = {k: v for k, v in res.items() if k not in ("keyframe_rgb", "evidence_rgb")}
        out["keyframe_jpg_base64"] = Predictor.encode_jpg(res["keyframe_rgb"])
        return out
