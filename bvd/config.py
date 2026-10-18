"""Global configuration."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- data ---
    labels_path: Path = ROOT / "data" / "labels.xlsx"
    cache_dir: Path = ROOT / "cache" / "features"
    artifacts_dir: Path = ROOT / "artifacts"
    model_path: Path = ROOT / "artifacts" / "model.joblib"
    video_exts: tuple = (".mov", ".mp4", ".avi", ".mkv", ".m4v")

    # --- sampling ---
    sample_fps: float = 2.0          # кадров в секунду на анализ
    max_side: int = 640              # даунскейл кадров при чтении

    # --- backbones ---
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    yolo_weights: str = "yolov8s.pt"
    yolo_conf: float = 0.25

    # --- labels -> frame targets ---
    pos_radius: float = 1.5          # |t - ts| <= r  -> кадр "нарушение"
    ignore_radius: float = 5.0       # r < |t - ts| <= ignore -> не используем
    neg_weight_in_pos_video: float = 0.3  # вес "нормальных" кадров из видео с нарушением

    # --- model ---
    n_pca: int = 32
    context_k: int = 3               # окно контекста = ±k кадров (±1.5 c при 2 fps)
    C: float = 0.05
    smooth_k: int = 2                # сглаживание вероятностей (±k кадров)
    default_threshold: float = 0.5
    # Порог на инференсе. Переопределяет порог, сохранённый в model.joblib.
    # Пропуск нарушения для банка недопустим (FAQ п.16), поэтому порог снижен до 0.5.
    # На OOF: 0.50 и 0.45 дают одинаковый recall (15/22), 0.45 добавляет лишний FP.
    # Можно поменять без переобучения: переменная окружения BVD_THRESHOLD=0.45
    inference_threshold: float = float(os.environ.get("BVD_THRESHOLD", "0.5"))


CFG = Config()

# Что видно на кадре и почему это может указывать на нарушение (без ссылок на конкретные пункты,
# которые нельзя подтвердить по видео).
VIOLATION_EXPLANATIONS = {
    "UNATTENDED_CASH_DESK": "кассир покидает рабочее место, касса и ценности на столе остаются без присмотра; "
                            "ценности должны находиться под контролем ответственного работника.",
    "THIRD_PARTY_INTRUSION": "в кассовой зоне за рабочим местом кассира находится посторонний человек; "
                             "доступ в кассовое помещение ограничен уполномоченными работниками.",
    "INCASSATION_BAG_VIOLATION": "действия с инкассаторской сумкой выполняются с отступлением от обычного порядка "
                                 "приёма/передачи (вскрытие или передача вне установленной процедуры).",
    "SAFE_AND_STORAGE_VIOLATION": "наличные/ценности перемещаются у сейфа или шкафа вне зоны пересчёта; "
                                  "хранение и выдача ценностей должны происходить в установленном порядке.",
    "DESK_CASH_HANDLING_VIOLATION": "наличные передаются из рук в руки без видимого пересчёта либо убираются "
                                    "из поля зрения камеры; операции с наличными выполняются открыто и с пересчётом.",
}

# Типы нарушений (из описания организаторов)
VIOLATION_TYPES = {
    "UNATTENDED_CASH_DESK": "Касса/сейф без присмотра (кассир покинул место)",
    "THIRD_PARTY_INTRUSION": "Посторонний в кассовой зоне",
    "INCASSATION_BAG_VIOLATION": "Нарушение работы с инкассаторской сумкой",
    "SAFE_AND_STORAGE_VIOLATION": "Операции с наличными у сейфа/шкафа вне зоны пересчёта",
    "DESK_CASH_HANDLING_VIOLATION": "Передача наличных из рук в руки без пересчёта / сокрытие",
}

VIDEO_TYPE = {
    **{v: "UNATTENDED_CASH_DESK" for v in ["X-4", "X-11", "X-15", "X-22", "X-24", "X-26", "X-28"]},
    **{v: "THIRD_PARTY_INTRUSION" for v in ["X-18", "X-30", "X-34", "X-38"]},
    **{v: "INCASSATION_BAG_VIOLATION" for v in ["X-17", "X-40"]},
    **{v: "SAFE_AND_STORAGE_VIOLATION" for v in ["X-13", "X-20", "X-32", "X-33"]},
    **{v: "DESK_CASH_HANDLING_VIOLATION" for v in ["X-1", "X-2", "X-44", "X-46", "X-47"]},
}

# Текстовые подсказки для zero-shot признаков CLIP
PROMPTS = [
    "a cashier sitting at a desk and working",
    "a woman sitting alone at an office desk",
    "an empty office desk with nobody sitting at it",
    "a woman standing up from her desk",
    "a woman walking away from the desk",
    "a person walking out through a glass door",
    "a woman standing next to a gray metal safe",
    "a person opening a cabinet",
    "banknotes spread on a desk",
    "hands counting cash money",
    "a money counting machine on a desk",
    "two people sitting across a desk from each other",
    "three people in a small office room",
    "a man standing behind a seated woman",
    "a man in a white shirt walking in an office",
    "a man in black clothes walking in an office",
    "a large bag on the desk",
    "an open bag on a table",
    "a person using a mobile phone",
    "a person hiding something in their lap",
]

# COCO-классы для YOLO
COCO_PERSON = 0
COCO_BAGS = (24, 26, 28)   # backpack, handbag, suitcase
COCO_PHONE = 67
COCO_BOOK = 73
YOLO_CLASSES = [COCO_PERSON, *COCO_BAGS, COCO_PHONE, COCO_BOOK]
