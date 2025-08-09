from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json

# --- project imports ---
from src.infer_classifier import BiasClassifier
from src.explain import Explainer


# Neutralizer is optional – try to import it; if missing we noop gracefully.
try:
    from src.neutralizer import Neutralizer
    _HAS_NEUTRALIZER = True
except Exception:
    _HAS_NEUTRALIZER = False

# Optional: config threshold
_DEF_THRESHOLD = 0.5
try:
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f) or {}
        _DEF_THRESHOLD = float(cfg.get("threshold", _DEF_THRESHOLD))
except Exception:
    cfg = {}

app = FastAPI(title="Bias Detection & Neutralization API", version="1.0.0")

# Serve /web if present
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/", response_class=HTMLResponse)
def root_page():
    index_path = os.path.join("web", "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
      <head><title>Bias API</title></head>
      <body style="font-family:system-ui,sans-serif">
        <h2>Bias Detection & Neutralization API</h2>
        <p>Open <a href="/docs">/docs</a> to try the API.</p>
      </body>
    </html>
    """

# ---------- load models ----------
MODEL_DIR = "models/bias_classifier"  # we trained & saved here

classifier = BiasClassifier(model_dir=MODEL_DIR, threshold=_DEF_THRESHOLD)
explainer = Explainer(model_dir=MODEL_DIR)

neutralizer = None
if _HAS_NEUTRALIZER:
    try:
        # If you trained/saved one, set its folder here; else the Neutralizer class may download a small model on first use.
        neutralizer = Neutralizer(model_dir="models/neutralizer")
    except Exception:
        neutralizer = None


# ---------- request / response ----------
class PredictIn(BaseModel):
    text: str


@app.post("/predict")
def predict(inp: PredictIn):
    text = (inp.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is required."})

    # 1) classify
    cls = classifier.predict(text)  # {bias_type, confidence, probs}

    # 2) neutralize (optional)
    neutralized = text
    if neutralizer:
        try:
            if hasattr(neutralizer, "neutralize"):
                neutralized = neutralizer.neutralize(text)
            elif hasattr(neutralizer, "rewrite"):
                neutralized = neutralizer.rewrite(text)
        except Exception as e:
            print("[api] neutralizer error:", e)
            neutralized = text

    # 3) highlights
    try:
        highlights = explainer.highlight(text, top_k=8)
    except Exception as e:
        print("[api] explainer error:", e)
        highlights = []

    return {
        "original_text": text,
        "bias_type": cls.get("bias_type", "no_bias"),
        "confidence": cls.get("confidence", 0.0),
        "probs": cls.get("probs", {}),
        "neutralized_text": neutralized,
        "highlights": highlights,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR, "threshold": _DEF_THRESHOLD}
