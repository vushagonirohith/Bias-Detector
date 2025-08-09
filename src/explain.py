import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients


class Explainer:
    """
    Token/word-level attribution using Integrated Gradients on input embeddings,
    with a safe fallback to plain gradients if IG yields near-zero attributions.
    Returns word-level spans: [{text, start, end, score}, ...]
    """

    def __init__(self, model_dir: str = "models/bias_classifier", max_length: int = 128):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.labels = getattr(self.model.config, "id2label", None)
        self.max_length = max_length

    def _forward_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: no no_grad() here — IG needs gradients
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

    def _drop_specials(self, tokens, scores, offsets):
        # Drop special tokens via offset mapping: specials usually have (0,0)
        out = [(t, s, off) for (t, s, off) in zip(tokens, scores, offsets) if not (off[0] == 0 and off[1] == 0)]
        return out

    def _merge_subwords_to_words(self, tokens, token_scores, offsets, top_k: int | None):
        """
        Merge BPE/subword pieces into word-level spans using tokenizer offsets.
        Works well for RoBERTa/DistilRoBERTa where word starts are marked with 'Ġ'.
        """
        pairs = self._drop_specials(tokens, token_scores, offsets)

        word_spans = []  # each: {text, start, end, score}
        curr = None

        for tok, score, (start, end) in pairs:
            piece = tok
            # For RoBERTa-family, a word start is usually prefixed with 'Ġ'
            is_word_start = piece.startswith("Ġ")
            clean = piece.lstrip("Ġ")

            if curr is None or is_word_start:
                if curr:
                    word_spans.append(curr)
                curr = {"text": clean, "start": int(start), "end": int(end), "score": float(score)}
            else:
                # continuation of the previous word
                curr["text"] += clean
                curr["end"] = int(end)
                curr["score"] += float(score)

        if curr:
            word_spans.append(curr)

        # sort by importance
        word_spans.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            word_spans = word_spans[:top_k]
        return word_spans

    def highlight(self, text: str, target_idx: int | None = None, top_k: int = 10, n_steps: int = 32):
        # 1) Tokenize with offsets for word spans
        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]            # [1, seq]
        attention_mask = enc["attention_mask"]  # [1, seq]
        offsets = enc["offset_mapping"].squeeze(0).tolist()  # [(start, end), ...]

        # 2) Choose target class
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            if target_idx is None:
                target_idx = int(torch.softmax(logits, dim=-1).argmax())

        # 3) Build embeddings with grad
        emb_layer = self.model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)     # [1, seq, hidden]
        inputs_embeds.requires_grad_()

        # 4) Integrated Gradients on embeddings
        ig = IntegratedGradients(lambda e, m: self._forward_embeds(e, m)[:, target_idx])
        attributions = ig.attribute(
            inputs=inputs_embeds,
            additional_forward_args=(attention_mask,),
            baselines=torch.zeros_like(inputs_embeds),
            n_steps=n_steps
        )  # [1, seq, hidden]

        ig_scores = attributions.abs().sum(dim=-1).squeeze(0)  # [seq]

        # 5) Fallback to plain grads if IG is degenerate
        if float(ig_scores.abs().sum()) == 0.0:
            self.model.zero_grad(set_to_none=True)
            inputs_embeds.grad = None
            for p in self.model.parameters():
                p.requires_grad_(False)
            inputs_embeds.requires_grad_(True)

            logits = self._forward_embeds(inputs_embeds, attention_mask)
            loss = logits[0, target_idx]
            loss.backward()

            grads = inputs_embeds.grad  # [1, seq, hidden]
            token_scores = grads.abs().sum(dim=-1).squeeze(0).tolist()
        else:
            token_scores = ig_scores.tolist()

        # 6) Convert IDs to tokens
        tokens = self.tok.convert_ids_to_tokens(input_ids.squeeze(0))

        # 7) Merge to word spans & return
        word_spans = self._merge_subwords_to_words(tokens, token_scores, offsets, top_k=top_k)
        return word_spans
