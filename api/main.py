"""REST API.  uvicorn api.main:app --host 0.0.0.0 --port 8000"""
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bvd.config import CFG  # noqa: E402
from bvd.predictor import Predictor  # noqa: E402

app = FastAPI(title="Bank cash-desk violation detector", version="1.0")
_pred: Predictor | None = None


def predictor() -> Predictor:
    global _pred
    if _pred is None:
        _pred = Predictor()
    return _pred


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG.model_path.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mov").suffix.lower()
    if suffix not in CFG.video_exts:
        raise HTTPException(400, f"Неподдерживаемый формат {suffix}. Допустимо: {CFG.video_exts}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        res = predictor().predict(path)
        out = Predictor.to_json(res)
        out["filename"] = file.filename
        return out
    except IOError as e:
        raise HTTPException(422, str(e))
    finally:
        Path(path).unlink(missing_ok=True)
