import os, pandas as pd, numpy as np, json, argparse, re
from sklearn.model_selection import train_test_split
from src.utils import write_jsonl, load_config

BIAS_TYPES = [
    "race-color","gender","sexual-orientation","religion","age",
    "nationality","disability","physical-appearance","socioeconomic"
]

def normalize_text(s: str) -> str:
    s = s.strip().replace("\u200b", "")
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--crows", default="data/crowspairs.csv")
    parser.add_argument("--neutral_extra", default="data/neutral_extra.txt")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    labels = cfg["labels"]
    label2id = {l:i for i,l in enumerate(labels)}

    df = pd.read_csv(args.crows)
    df = df.rename(columns={c:c.strip() for c in df.columns})
    if not {"sent_more","sent_less","bias_type"}.issubset(df.columns):
        raise ValueError("crowspairs.csv must contain: sent_more, sent_less, bias_type")

    # Biased samples = sent_more with its bias_type
    biased = pd.DataFrame({
        "text": df["sent_more"].astype(str).map(normalize_text),
        "label": df["bias_type"].astype(str)
    })
    biased = biased[biased["label"].isin(BIAS_TYPES)].dropna()

    # No-bias pool from sent_less + optional neutral_extra
    no_bias_rows = df["sent_less"].astype(str).map(normalize_text).tolist()
    if os.path.isfile(args.neutral_extra):
        with open(args.neutral_extra, "r", encoding="utf-8") as f:
            extra = [normalize_text(x) for x in f if x.strip()]
            no_bias_rows.extend(extra)
    no_bias = pd.DataFrame({"text": no_bias_rows})
    no_bias["label"] = "no_bias"

    # ---------- BALANCE THE DATASET ----------
    # 1) Equalize biased classes to the smallest biased class count
    if len(biased) == 0:
        raise ValueError("No biased rows found after filtering. Check your CSV columns/values.")
    counts = biased["label"].value_counts()
    per_class = int(counts.min())  # conservative choice
    balanced_biased = (
        biased.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), per_class), random_state=42))
              .reset_index(drop=True)
    )

    # 2) Downsample no_bias to match per_class
    if len(no_bias) >= per_class:
        balanced_no_bias = no_bias.sample(n=per_class, random_state=42).reset_index(drop=True)
    else:
        balanced_no_bias = no_bias

    # 3) Merge, dedupe, filter empties
    all_df = pd.concat([balanced_biased, balanced_no_bias], ignore_index=True)
    all_df.drop_duplicates(subset=["text"], inplace=True)
    all_df = all_df[all_df["text"].str.len() > 0]

    print("Class counts after balancing:")
    print(all_df["label"].value_counts().sort_index())

    # Map labels & stratified split
    all_df["label_id"] = all_df["label"].map(label2id)
    train_df, temp_df = train_test_split(all_df, test_size=0.2, stratify=all_df["label_id"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_df[["text","label_id"]].to_dict(orient="records"))
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"),   val_df[["text","label_id"]].to_dict(orient="records"))
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"),  test_df[["text","label_id"]].to_dict(orient="records"))

    with open("src/label_map.json","w") as f:
        json.dump({
            "labels": labels,
            "label2id": {l:i for i,l in enumerate(labels)},
            "id2label": {i:l for i,l in enumerate(labels)}
        }, f, indent=2)

    print("Data prepared:", {k: len(v) for k,v in {"train":train_df, "val":val_df, "test":test_df}.items()})

if __name__ == "__main__":
    main()
