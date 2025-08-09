import json, os, random
import numpy as np
import torch
import yaml
from typing import List, Dict
from sklearn.metrics import log_loss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def temperature_scale(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / T

def find_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    best_T, best_nll = 1.0, 1e9
    for T in np.linspace(0.5, 3.0, 26):
        with torch.no_grad():
            scaled = temperature_scale(logits, T)
            probs = torch.softmax(scaled, dim=-1).cpu().numpy()
        nll = log_loss(labels.cpu().numpy(), probs, labels=np.unique(labels.cpu().numpy()))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T
