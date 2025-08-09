import argparse, os, json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default="models/bias_classifier")
    ap.add_argument("--base", default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", default="models/bias_classifier_merged")
    ap.add_argument("--label_map", default="src/label_map.json")
    args = ap.parse_args()

    # Load label mapping (so the merged model keeps proper id2label/label2id)
    if os.path.isfile(args.label_map):
        lm = json.load(open(args.label_map, "r"))
        id2label = {int(k): v for k, v in lm.get("id2label", {}).items()}
        num_labels = len(id2label) if id2label else None
    else:
        id2label, num_labels = None, None

    # Build base model with correct num_labels if we know them
    if num_labels is None:
        # Try to read adapter config for labels
        try:
            acfg = AutoConfig.from_pretrained(args.adapter_dir)
            if hasattr(acfg, "id2label") and acfg.id2label:
                id2label = {int(k): v for k, v in acfg.id2label.items()}
                num_labels = len(id2label)
        except Exception:
            pass

    if num_labels is None:
        raise RuntimeError("Could not determine num_labels. Ensure src/label_map.json exists.")

    label2id = {v: k for k, v in id2label.items()}

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    # Merge LoRA adapter into the base weights and unload PEFT wrappers
    model = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)

    # Prefer tokenizer from adapter_dir (it carries any special tokens); else fall back to base
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out_dir)

    # Also save label map for safety
    if id2label:
        with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
            json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"Merged model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
