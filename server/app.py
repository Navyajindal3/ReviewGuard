import os
import numpy as np
import joblib
from typing import Any, List, Union

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import custom_vec

# -------- Config --------
PIPELINE_PATH   = os.getenv("PIPELINE_PATH",   "pipeline.pkl")       # Recommended artifact
MODEL_PATH      = os.getenv("MODEL_PATH",      "model.pkl")          # Fallback
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")     # Fallback
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.65"))
FAKE_LABEL = os.getenv("FAKE_LABEL", "1").lower()  # used to pick the "positive" class column if needed

# -------- App --------
app = FastAPI(title="Fake Review Guard API", version="0.3.0")  # you can set debug=True during dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Models --------
class PredictRequest(BaseModel):
    texts: List[str]

pipeline = None
vectorizer = None
model = None

def _unwrap_last_estimator(est):
    """Return the final estimator for classes_ lookup."""
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import VotingClassifier
    except Exception:
        return est

    # Pipeline -> last step
    if "sklearn.pipeline" in str(type(est)):
        try:
            return est.steps[-1][1]
        except Exception:
            return est

    # CalibratedClassifierCV -> base_estimator_
    if hasattr(est, "base_estimator_"):
        return est.base_estimator_

    # VotingClassifier -> itself (has classes_), or pick a base model
    if "VotingClassifier" in str(type(est)):
        return est

    return est

def _positive_col_index(est) -> int:
    """
    Determine which probability/score column corresponds to the 'fake' class.
    Tries:
      - match FAKE_LABEL env (default '1')
      - match 'fake' by name
      - if binary, use the last column
    """
    est2 = _unwrap_last_estimator(est)
    classes = getattr(est2, "classes_", None)
    if classes is None:
        return -1

    cls_str = [str(c).lower() for c in classes]
    # explicit FAKE_LABEL
    if FAKE_LABEL in cls_str:
        return cls_str.index(FAKE_LABEL)
    # try "fake" by name
    if "fake" in cls_str:
        return cls_str.index("fake")
    # typical binary setups: pick the last column as "positive"
    if len(cls_str) == 2:
        return 1
    # fallback: last column
    return len(cls_str) - 1

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _to_text_list(payload: Any) -> List[str]:
    """Accept {texts:[...]}, {text:"..."}, raw list, or raw string."""
    if isinstance(payload, PredictRequest):
        return [str(t) for t in payload.texts]
    if isinstance(payload, list):
        return [str(t) for t in payload]
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        if "texts" in payload and isinstance(payload["texts"], list):
            return [str(t) for t in payload["texts"]]
        if "text" in payload:
            return [str(payload["text"])]
    raise HTTPException(status_code=400, detail='Invalid body. Send { "texts": ["..."] }')

def _probs_from_estimator(est, X_or_texts):
    """
    Return probabilities for the "fake" class across many estimator types.
    Handles:
      - Pipeline (vectorizer inside) OR separate vectorizer+model
      - predict_proba with class column selection
      - decision_function 1D or 2D
      - predict fallback mapping to {0,1}
    """
    # If est is a pipeline and we passed raw texts, it will vectorize internally.
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_or_texts)
        proba = np.asarray(proba)
        if proba.ndim == 1:  # rare
            return np.clip(proba.astype(float), 0.0, 1.0)
        col = _positive_col_index(est)
        if 0 <= col < proba.shape[1]:
            return np.clip(proba[:, col].astype(float), 0.0, 1.0)
        # fallback: last column
        return np.clip(proba[:, -1].astype(float), 0.0, 1.0)

    if hasattr(est, "decision_function"):
        scores = est.decision_function(X_or_texts)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            return np.clip(_sigmoid(scores), 0.0, 1.0)
        # 2D: choose the positive column or collapse to binary margin
        if scores.shape[1] == 2:
            # score for positive minus negative -> then sigmoid
            margin = scores[:, 1] - scores[:, 0]
            return np.clip(_sigmoid(margin), 0.0, 1.0)
        col = _positive_col_index(est)
        if 0 <= col < scores.shape[1]:
            return np.clip(_sigmoid(scores[:, col]), 0.0, 1.0)
        return np.clip(_sigmoid(scores[:, -1]), 0.0, 1.0)

    # Last resort: use predict -> map labels to {0,1}
    preds = est.predict(X_or_texts)
    out = []
    for y in preds:
        yl = str(y).lower()
        if yl in {"1", "true", "yes", "fake", "spam"}:
            out.append(1.0)
        elif yl in {"0", "false", "no", "real", "genuine"}:
            out.append(0.0)
        elif isinstance(y, (int, float)) and y in (0, 1):
            out.append(float(y))
        else:
            # unknown label names: treat the last class as positive
            out.append(1.0 if yl == FAKE_LABEL else 0.0)
    return np.asarray(out, dtype=float)

# -------- Lifecycle --------
@app.on_event("startup")
def load_artifacts():
    global pipeline, vectorizer, model
    if os.path.exists(PIPELINE_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        return
    # separate artifacts path
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    # quick dimension sanity check if available
    try:
        nv = len(getattr(vectorizer, "get_feature_names_out", lambda: vectorizer.vocabulary_.keys())())
    except Exception:
        nv = None
    try:
        nm = getattr(model, "n_features_in_", None)
        if nm is None and hasattr(model, "coef_"):
            nm = model.coef_.shape[1]
    except Exception:
        nm = None
    if nv and nm and nv != nm:
        raise RuntimeError(f"Vectorizer/model mismatch: vectorizer={nv} features, model expects {nm}")

# -------- Routes --------
@app.get("/health")
def health():
    mode = "pipeline" if pipeline is not None else "vec+model"
    return {"status": "ok", "mode": mode, "threshold": DEFAULT_THRESHOLD}

@app.get("/")
def root():
    return {"message": "Fake Review Guard API – see /docs", "routes": ["/health", "/predict"]}

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

@app.post("/predict")
def predict(payload: Union[PredictRequest, List[str], str, dict] = Body(...)):
    texts = _to_text_list(payload)
    try:
        if pipeline is not None:
            proba = _probs_from_estimator(pipeline, texts)
        else:
            X = vectorizer.transform(texts)
            proba = _probs_from_estimator(model, X)
        results = []
        for p in np.asarray(proba).ravel().tolist():
            label = "likely fake" if float(p) >= DEFAULT_THRESHOLD else "likely genuine"
            results.append({"proba_fake": round(float(p), 4), "label": label})
        return {"results": results, "threshold": DEFAULT_THRESHOLD}
    except Exception as e:
        # Return a readable JSON error instead of blank 500
        raise HTTPException(status_code=500, detail=f"inference_error: {type(e).__name__}: {e}")

# (Optional) debug helper to see what body the server receives
@app.post("/echo")
async def echo(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = await req.body()
    return {"content_type": req.headers.get("content-type"), "body": body}
