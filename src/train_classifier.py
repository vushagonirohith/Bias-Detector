import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

# === PRINT 1: Start of script ===
print("[train_classifier] Starting…")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"weighted_f1": f, "macro_f1": mf1, "precision": p, "recall": r}

class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (robust to new HF kwargs)."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with train.jsonl/val.jsonl/test.jsonl")
    ap.add_argument("--out_dir",  required=True, help="Where to save the trained model")
    ap.add_argument("--model_name", default="distilroberta-base")  # faster & learns on CPU
    ap.add_argument("--epochs", type=int, default=8)               # longer, still OK on CPU
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    # --- Load label map so model saves with real names ---
    lm = json.load(open("src/label_map.json", "r"))
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    tok = AutoTokenizer.from_pretrained(args.model_name)

    # Load JSONL splits (fields: text, label_id)
    train_raw = load_dataset("json", data_files=os.path.join(args.data_dir, "train.jsonl"))["train"]
    val_raw   = load_dataset("json", data_files=os.path.join(args.data_dir, "val.jsonl"))["train"]
    test_raw  = load_dataset("json", data_files=os.path.join(args.data_dir, "test.jsonl"))["train"]

    # === PRINT 2: Dataset sizes ===
    print(f"[train_classifier] Dataset sizes -> Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    # Shorter sequences help a lot on CPU
    def tok_fn(batch):
        enc = tok(batch["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = batch["label_id"]
        return enc

    train_ds = train_raw.map(tok_fn, batched=True)
    val_ds   = val_raw.map(tok_fn, batched=True)
    test_ds  = test_raw.map(tok_fn, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        cols.append("token_type_ids")
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    # --- Class weights (dataset is balanced now, but harmless to keep) ---
    y_train = np.array(train_raw["label_id"])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[train_classifier] Class weights -> {class_weights.tolist()}")

    # --- Model with correct label mapping ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # CPU-friendly: eval/save per epoch, higher LR for small model
    args_tr = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-5,                # higher LR for small model
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=False
    )

    trainer = WeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # === PRINT 3: Before training ===
    print("[train_classifier] Launching Trainer.train()…")

    trainer.train()

    # Save final model + tokenizer
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # Final test report
    test_logits = trainer.predict(test_ds).predictions
    test_preds = test_logits.argmax(-1)
    rep = classification_report(
        test_raw["label_id"], test_preds,
        target_names=[id2label[i] for i in range(num_labels)],
        digits=4, zero_division=0
    )
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep)
    print(f"[train_classifier] Training complete. Model saved to {args.out_dir}")

if __name__ == "__main__":
    main()
