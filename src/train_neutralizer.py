import os, argparse, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/neutral_pairs.csv")  # columns: biased, neutral
    parser.add_argument("--out_dir", default="models/neutralizer")
    parser.add_argument("--base_model", default="google/flan-t5-base")
    args = parser.parse_args()

    if not os.path.isfile(args.pairs):
        raise FileNotFoundError(f"{args.pairs} not found. Provide parallel data or skip training.")

    df = pd.read_csv(args.pairs)
    if not {"biased","neutral"}.issubset(df.columns):
        raise ValueError("neutral_pairs.csv must have columns: biased, neutral")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    def prep(batch):
        inputs = [f"Rewrite the text to remove stereotypes or social bias while preserving meaning and tone. Keep names and facts. Text: {t}" for t in batch["biased"]]
        mi = tok(inputs, truncation=True, max_length=256, padding="max_length")
        with tok.as_target_tokenizer():
            labels = tok(batch["neutral"], truncation=True, max_length=256, padding="max_length")
        mi["labels"] = labels["input_ids"]
        return mi

    ds = Dataset.from_pandas(df).map(prep, batched=True).remove_columns([c for c in df.columns if c not in []])

    args_tr = TrainingArguments(
        output_dir="models/neutralizer_ckpt",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        save_strategy="epoch",
        logging_steps=50,
        fp16=True
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, tokenizer=tok)
    trainer.train()
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Neutralizer saved to", args.out_dir)

if __name__ == "__main__":
    main()
