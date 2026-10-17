"""Batch-предсказание по папке (например, для X-6, X-7, X-9, X-36, X-41, X-42 без разметки).

    python scripts/predict_folder.py --videos D:/videos --out artifacts/predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bvd.labels import find_videos  # noqa: E402
from bvd.predictor import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="artifacts/predictions.csv")
    args = ap.parse_args()
    p = Predictor()
    rows = []
    for vid, path in find_videos(Path(args.videos)).items():
        r = p.predict(path)
        rows.append({"video": vid, "file": path.name, "verdict": r["verdict"], "probability": r["probability"],
                     "timestamp_sec": r["timestamp_sec"], "type": r["violation_type"]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print("saved", args.out)


if __name__ == "__main__":
    main()
