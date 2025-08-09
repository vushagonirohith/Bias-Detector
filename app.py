import os
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: Captum for Integrated Gradients
_HAS_CAPTUM = True
try:
    from captum.attr import IntegratedGradients
except Exception:
    _HAS_CAPTUM = False

# Optional neutralizer (lazy)
_NEUT = {"tok": None, "model": None}
_NEUT_BASE = os.environ.get("NEUTRALIZER_MODEL", "google/flan-t5-small")

MODEL_DIR = os.environ.get("MODEL_DIR", "models/bias_classifier")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Load classifier once (cached)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
clf = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
clf.eval()

ID2LABEL = getattr(clf.config, "id2label", None)
if isinstance(ID2LABEL, dict):
    ID2LABEL = {int(k): v for k, v in ID2LABEL.items()}
else:
    ID2LABEL = {i: f"LABEL_{i}" for i in range(clf.config.num_labels)}
LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]

# Tiny in-memory cache for repeated texts
_PRED_CACHE: dict[str, dict] = {}

app = Flask(__name__)

# ---------- Explainer ----------
def _forward_embeds(inputs_embeds, attention_mask):
    return clf(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

def _merge_subwords(tokens, scores, offsets, top_k=8):
    # Drop specials via offsets (0,0)
    items = [(t, s, off) for t, s, off in zip(tokens, scores, offsets) if not (off[0] == 0 and off[1] == 0)]
    spans, curr = [], None
    for t, s, (st, en) in items:
        is_start = t.startswith("Ġ")
        piece = t.lstrip("Ġ")
        if curr is None or is_start:
            if curr: spans.append(curr)
            curr = {"text": piece, "start": int(st), "end": int(en), "score": float(s)}
        else:
            curr["text"] += piece
            curr["end"] = int(en)
            curr["score"] += float(s)
    if curr: spans.append(curr)
    spans.sort(key=lambda x: x["score"], reverse=True)
    return spans[:top_k] if top_k is not None else spans

def explain_text(text, max_length=128, top_k=8, n_steps=32):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"].squeeze(0).tolist()

    with torch.no_grad():
        logits = clf(input_ids=input_ids, attention_mask=attention_mask).logits
        target_idx = int(F.softmax(logits, dim=-1).argmax())

    emb_layer = clf.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids)
    inputs_embeds.requires_grad_()

    token_scores = None
    if _HAS_CAPTUM:
        ig = IntegratedGradients(lambda e, m: _forward_embeds(e, m)[:, target_idx])
        attributions = ig.attribute(
            inputs=inputs_embeds,
            additional_forward_args=(attention_mask,),
            baselines=torch.zeros_like(inputs_embeds),
            n_steps=n_steps
        )
        ig_scores = attributions.abs().sum(dim=-1).squeeze(0)
        if float(ig_scores.abs().sum()) != 0.0:
            token_scores = ig_scores.tolist()

    if token_scores is None:
        clf.zero_grad(set_to_none=True)
        inputs_embeds.grad = None
        for p in clf.parameters(): p.requires_grad_(False)
        inputs_embeds.requires_grad_(True)
        logits2 = _forward_embeds(inputs_embeds, attention_mask)
        loss = logits2[0, target_idx]
        loss.backward()
        grads = inputs_embeds.grad
        token_scores = grads.abs().sum(dim=-1).squeeze(0).tolist()

    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0))
    return _merge_subwords(tokens, token_scores, offsets, top_k=top_k)

# ---------- Neutralizer ----------
def neutralize_text(text: str) -> str:
    try:
        if _NEUT["tok"] is None or _NEUT["model"] is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            _NEUT["tok"] = AutoTokenizer.from_pretrained(_NEUT_BASE)
            _NEUT["model"] = AutoModelForSeq2SeqLM.from_pretrained(_NEUT_BASE)
            _NEUT["model"].eval()
        prompt = (
            "Rewrite to be neutral, fair, and bias-free while preserving intended meaning and tone. "
            "Avoid stereotypes.\nText: " + text
        )
        x = _NEUT["tok"](prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            out = _NEUT["model"].generate(
                **x, max_new_tokens=64, num_beams=4, length_penalty=0.9, early_stopping=True
            )
        return _NEUT["tok"].decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print("[neutralizer] error:", e)
        return text

# ---------- Routes ----------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/examples")
def api_examples():
    examples = {
        "Gender": [
            "Women are too emotional for leadership.",
            "Men are naturally better at engineering.",
            "A woman should not drive alone at night."
        ],
        "Race/Color": [
            "Asians are naturally good at math.",
            "Dark-skinned people are less trustworthy."
        ],
        "Age": [
            "Older employees can’t learn new technology.",
            "Young people don’t understand responsibility."
        ],
        "Nationality": [
            "Indians can’t speak good English.",
            "Americans are all overweight."
        ],
        "Religion": [
            "Muslims are dangerous.",
            "Christians are more moral."
        ],
        "Disability": [
            "Disabled people can’t work as efficiently.",
            "Autistic children can’t study in normal schools."
        ],
        "Sexual orientation": [
            "Gay people should not adopt children.",
            "Bisexuals are just confused."
        ],
        "Appearance": [
            "Fat people are lazy.",
            "Short men can’t be leaders."
        ],
        "Socioeconomic": [
            "Poor people are criminals.",
            "Rich people are selfish."
        ]
    }
    return jsonify(examples)

@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required."}), 400

    # optional per-request threshold & neutralize flag
    threshold = float(data.get("threshold", THRESHOLD))
    do_neutralize = bool(data.get("neutralize", True))

    # cache hit?
    cache_key = f"{text}||{threshold}"
    cached = _PRED_CACHE.get(cache_key)
    if cached:
        resp = cached.copy()
    else:
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = clf(**enc).logits
            probs = F.softmax(logits, dim=-1)[0].tolist()

        probs_map = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_label = ID2LABEL[pred_idx]
        confidence = float(probs[pred_idx])
        bias_type = pred_label if confidence >= threshold else "no_bias"

        # highlights best-effort
        try:
            highlights = explain_text(text, max_length=128, top_k=8)
        except Exception as e:
            print("[explain] error:", e)
            highlights = []

        resp = {
            "original_text": text,
            "bias_type": bias_type,
            "confidence": round(confidence, 4),
            "probs": probs_map,
            "neutralized_text": "",   # filled below if requested
            "highlights": highlights
        }
        _PRED_CACHE[cache_key] = resp.copy()

    # neutralize lazily when requested
    if do_neutralize and not resp.get("neutralized_text"):
        resp["neutralized_text"] = neutralize_text(text)
    elif not do_neutralize:
        resp["neutralized_text"] = text

    return jsonify(resp)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_dir": MODEL_DIR, "threshold": THRESHOLD, "captum": _HAS_CAPTUM})

if __name__ == "__main__":
    # pip install flask transformers torch captum
    app.run(host="0.0.0.0", port=5000, debug=False)
