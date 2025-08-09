import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BiasClassifier:
    def __init__(self, model_dir="models/bias_classifier", threshold=0.5):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        # Get labels from model config first
        self.labels = self.model.config.id2label if getattr(self.model.config, "id2label", None) else {}
        # Fallback: load our label_map and override LABEL_i names
        try:
            lm = json.load(open("src/label_map.json", "r"))
            id2label_map = {int(k): v for k,v in lm.get("id2label", {}).items()}
            if id2label_map:
                # If model labels look like LABEL_0..., replace with real names
                sample_label = next(iter(self.labels.values())) if self.labels else "LABEL_0"
                if isinstance(sample_label, str) and sample_label.startswith("LABEL_"):
                    self.labels = id2label_map
                    # also push mapping into config so downstream code is clean
                    self.model.config.id2label = self.labels
                    self.model.config.label2id = {v:k for k,v in self.labels.items()}
        except Exception:
            pass

        # Temperature (optional)
        self.threshold = threshold
        self.temperature = 1.0
        calib = os.path.join(model_dir, "calibration.json")
        if os.path.isfile(calib):
            try:
                self.temperature = float(json.load(open(calib))["temperature"])
            except Exception:
                pass

    @torch.no_grad()
    def predict(self, text: str):
        x = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        logits = self.model(**x).logits / self.temperature
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top = int(probs.argmax())
        conf = float(probs[top])
        label_map = self.labels if self.labels else {i: f"LABEL_{i}" for i in range(len(probs))}
        bias_type = label_map[top] if conf >= self.threshold else "no_bias"
        return {
            "bias_type": bias_type,
            "confidence": round(conf, 4),
            "probs": {label_map[i]: float(p) for i, p in enumerate(probs)}
        }
