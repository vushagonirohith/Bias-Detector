from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class Neutralizer:
    def __init__(self, model_dir="models/neutralizer", base_model="google/flan-t5-base"):
        # Use fine-tuned model if present, else zero-shot fallback
        if os.path.isdir(model_dir) and any(os.scandir(model_dir)):
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        else:
            self.tok = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.model.eval()

    def neutralize(self, text: str) -> str:
        prompt = f"Rewrite the text to remove stereotypes or social bias while preserving meaning and tone. Keep names and facts. Text: {text}"
        x = self.tok(prompt, return_tensors="pt", truncation=True, max_length=256)
        y = self.model.generate(**x, max_new_tokens=128, num_beams=4, no_repeat_ngram_size=3)
        return self.tok.decode(y[0], skip_special_tokens=True)
