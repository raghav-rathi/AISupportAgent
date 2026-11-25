"""A large reranker on a GPU: the reference that small and hosted rerankers are measured against.

It needs the optional `lab` dependency group (torch and transformers) and a GPU, and it never runs
in production. The prompt format and the yes/no scoring follow the Qwen3-Reranker model card.
"""

from __future__ import annotations

import asyncio

SUPPORT_INSTRUCTION = (
    "Given a customer support question, retrieve help center passages that answer the question"
)

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenReranker:
    """Qwen3-Reranker through transformers. A score is the logit of "yes" minus the logit of "no"."""

    def __init__(
        self,
        model: str,
        instruction: str = SUPPORT_INSTRUCTION,
        *,
        max_length: int = 2048,
        batch_size: int = 20,
    ) -> None:
        # Imported here because only the lab dependency group installs them.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self._torch = torch
        self._instruction = instruction
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16).cuda().eval()
        self._yes = self._tokenizer.convert_tokens_to_ids("yes")
        self._no = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix = self._tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._room = max_length - len(self._prefix) - len(self._suffix)

    def _score(self, query: str, documents: list[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(documents), self._batch_size):
            texts = [
                f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {document}"
                for document in documents[start : start + self._batch_size]
            ]
            encoded = self._tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self._room,
            )
            encoded["input_ids"] = [
                self._prefix + ids + self._suffix for ids in encoded["input_ids"]
            ]
            batch = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
            batch = batch.to(self._model.device)
            with torch.no_grad():
                logits = self._model(**batch).logits[:, -1, :]
            scores += (logits[:, self._yes] - logits[:, self._no]).float().tolist()
        return scores

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return await asyncio.to_thread(self._score, query, documents)
