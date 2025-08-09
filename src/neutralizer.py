from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, os

class Neutralizer:
    def __init__(self, model_dir: str | None = None, base: str = "google/flan-t5-small"):
        # lightweight, CPU-friendly
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base)
        self.model.eval()

    @torch.inference_mode()
    def neutralize(self, text: str) -> str:
        prompt = f"Rewrite to be neutral, fair, and bias-free. Keep meaning. Text: {text}"
        x = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        out = self.model.generate(
            **x,
            max_new_tokens=64,
            num_beams=4,
            length_penalty=0.9,
            early_stopping=True
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # Optional alias
    def rewrite(self, text: str) -> str:
        return self.neutralize(text)
