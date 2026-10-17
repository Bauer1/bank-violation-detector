"""Извлечение признаков + leave-one-video-out валидация + обучение финальной модели.

    python scripts/train.py --videos "D:/hackathon/videos"
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bvd.config import CFG  # noqa: E402
from bvd.features import FeatureExtractor, cached_features  # noqa: E402
from bvd.labels import find_videos, load_labels  # noqa: E402
from bvd.model import ViolationModel  # noqa: E402


def pick_threshold(scores, y):
    """Порог, максимизирующий balanced accuracy на OOF-скорах (середина плато)."""
    cands = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best, best_ts = -1, []
    for t in cands:
        pred = scores >= t
        tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        tnr = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        ba = (tpr + tnr) / 2
        if ba > best + 1e-9:
            best, best_ts = ba, [t]
        elif abs(ba - best) < 1e-9:
            best_ts.append(t)
    lo = best_ts[0]
    below = scores[scores < lo]
    return float((lo + below.max()) / 2) if len(below) else float(lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="папка с видео (.mov)")
    ap.add_argument("--labels", default=str(CFG.labels_path))
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    videos = find_videos(Path(args.videos))
    ids = [v for v in labels if v in videos]
    missing = [v for v in labels if v not in videos]
    unlabeled = [v for v in videos if v not in labels]
    print(f"Размечено: {len(labels)}, найдено видео: {len(videos)}, используем: {len(ids)}")
    if missing:
        print("  нет файла для:", missing)
    if unlabeled:
        print("  без разметки (не участвуют в обучении):", unlabeled)

    ext = {}

    def factory():
        if "x" not in ext:
            ext["x"] = FeatureExtractor()
            print("  backbones loaded on", ext["x"].device)
        return ext["x"]

    feats = {}
    for i, v in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] features {v} ({videos[v].name})", flush=True)
        feats[v] = cached_features(factory, videos[v])

    # ---------- leave-one-video-out ----------
    rows = []
    for v in ids:
        tr = [u for u in ids if u != v]
        m = ViolationModel().fit([feats[u] for u in tr], [labels[u] for u in tr])
        r = m.predict(feats[v])
        lab = labels[v]
        rows.append({"video": v, "y": int(lab.violation), "ts_true": lab.ts, "type_true": lab.vtype,
                     "score": r["score"], "ts_pred": r["peak_time"], "type_pred": r["type"]})
        print(f"  LOVO {v:6s} y={int(lab.violation)} score={r['score']:.3f} peak={r['peak_time']:.1f}s "
              f"(true {lab.ts})")
    oof = pd.DataFrame(rows)
    y, s = oof["y"].values, oof["score"].values
    thr = pick_threshold(s, y)
    pred = (s >= thr).astype(int)
    pred_default = (s >= CFG.default_threshold).astype(int)
    oof["pred"] = pred

    tp = oof[(oof.y == 1) & (oof.pred == 1)]
    ts_err = (tp.ts_pred - tp.ts_true).abs()
    pos = oof[oof.y == 1]
    report = {
        "n_videos": len(oof), "threshold": thr,
        "threshold_tuned_on_oof": True,
        "accuracy_at_default_threshold": accuracy_score(y, pred_default),
        "precision_at_default_threshold": precision_score(y, pred_default, zero_division=0),
        "recall_at_default_threshold": recall_score(y, pred_default, zero_division=0),
        "f1_at_default_threshold": f1_score(y, pred_default, zero_division=0),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, s) if len(set(y)) > 1 else None,
        "timestamp_mae_sec_on_TP": float(ts_err.mean()) if len(tp) else None,
        "timestamp_within_3s_on_TP": float((ts_err <= 3).mean()) if len(tp) else None,
        "type_accuracy_on_positives": float((pos.type_pred == pos.type_true).mean()) if len(pos) else None,
    }
    print("\n===== Leave-one-video-out =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(CFG.artifacts_dir / "lovo_predictions.csv", index=False, encoding="utf-8-sig")
    (CFG.artifacts_dir / "cv_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    # ---------- final model ----------
    final = ViolationModel().fit([feats[v] for v in ids], [labels[v] for v in ids])
    final.threshold = thr
    joblib.dump(final, CFG.model_path)
    print(f"\nМодель сохранена: {CFG.model_path} (threshold={thr:.3f})")


if __name__ == "__main__":
    main()
