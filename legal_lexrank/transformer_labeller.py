"""Fine-tuned, non-generative transformer encoders for seven rhetorical roles.

One call processes one case. Every target token is kept in deterministic chunks;
only the immediately previous/following sentences are truncated for context.
"""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import ROLES


CHECKPOINT_FORMAT = "transformer-role-v1"
MODEL_IDS = {
    "inlegalbert": "law-ai/InLegalBERT",
    "deberta": "microsoft/deberta-v3-base",
}
MODEL_REVISIONS = {
    "inlegalbert": "b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a",
    "deberta": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
}
_MODEL_TYPES = {"inlegalbert": "bert", "deberta": "deberta-v2"}


def _load_tokenizer(encoder, cache_dir):
    # Slow DeBERTa uses SentencePiece directly, avoiding fast-tokenizer conversion.
    return AutoTokenizer.from_pretrained(
        MODEL_IDS[encoder], revision=MODEL_REVISIONS[encoder],
        cache_dir=cache_dir, use_fast=False, trust_remote_code=False,
    )


def _bounded_context(previous, following, budget, separator):
    """Keep the previous sentence's tail and the next sentence's beginning."""
    if budget <= 0:
        return []
    if not previous:
        return following[:budget]
    if not following:
        return previous[-budget:]
    if budget < 3:
        # There is no room for both neighbors and their separator.
        return previous[-budget:]
    available = budget - 1
    left = min(len(previous), (available + 1) // 2)
    right = min(len(following), available - left)
    left = min(len(previous), available - right)
    return previous[-left:] + [separator] + following[:right]


class TransformerRoleLabeller(nn.Module):
    """A fully trainable encoder, target-token mean pool, and fresh role head.

    ``tokenizer`` and ``backbone`` may be injected for small offline tests. Normal
    construction always uses the pinned unsupervised base encoder. Labels passed
    to ``make_examples`` use canonical role strings; ``None`` labels are omitted
    as targets while their text remains available as same-case context.
    """

    def __init__(self, encoder="inlegalbert", *, max_length=512,
                 target_chunk_size=384, dropout=0.1, batch_size=8,
                 cache_dir=None, tokenizer=None, backbone=None):
        super().__init__()
        if encoder not in MODEL_IDS:
            raise ValueError(f"Unknown transformer encoder: {encoder!r}")
        if not isinstance(max_length, int) or not 4 <= max_length <= 512:
            raise ValueError("max_length must be an integer between 4 and 512")
        if not isinstance(target_chunk_size, int) or not 1 <= target_chunk_size <= 384:
            raise ValueError("target_chunk_size must be an integer between 1 and 384")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between zero and one")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.encoder = encoder
        self.encoder_model = MODEL_IDS[encoder]
        self.encoder_revision = MODEL_REVISIONS[encoder]
        self.roles = ROLES
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.config = {
            "max_length": max_length, "target_chunk_size": target_chunk_size,
            "dropout": dropout, "batch_size": batch_size,
        }
        self.metadata = {}
        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(encoder, self.cache_dir)
        self._pair_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        if target_chunk_size + self._pair_special_tokens > max_length:
            raise ValueError("Target chunk and pair special tokens exceed max_length")
        if self.tokenizer.pad_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError("The tokenizer needs padding and separator tokens")
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(
            self.encoder_model, revision=self.encoder_revision, cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        if self.backbone.config.model_type != _MODEL_TYPES[encoder]:
            raise ValueError("Backbone architecture differs from the configured encoder")
        if getattr(self.backbone.config, "max_position_embeddings", max_length) < max_length:
            raise ValueError("max_length exceeds the encoder's position capacity")
        # Explicitly restore full fine-tuning if an injected model was frozen.
        self.backbone.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        self.to_roles = nn.Linear(self.backbone.config.hidden_size, len(self.roles))

    def make_examples(self, sentences, labels=None):
        """Encode one case, preserving every target token with weighted chunks."""
        sentences = list(sentences)
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise ValueError("Each sentence must be text")
        role_ids = {role: index for index, role in enumerate(self.roles)}
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(sentences):
                raise ValueError("Labels must align with sentences")
            if any(label is not None and label not in role_ids for label in labels):
                raise ValueError("Labels must use the canonical rhetorical roles")
        tokenized = []
        for sentence in sentences:
            ids = self.tokenizer.encode(sentence, add_special_tokens=False, truncation=False)
            if not ids:
                if self.tokenizer.unk_token_id is None:
                    raise ValueError("Empty sentences require a tokenizer unknown token")
                ids = [self.tokenizer.unk_token_id]
            tokenized.append(ids)
        examples = []
        chunk_size = self.config["target_chunk_size"]
        use_token_types = "token_type_ids" in self.tokenizer.model_input_names
        for sentence_index, target_ids in enumerate(tokenized):
            if labels is not None and labels[sentence_index] is None:
                continue
            previous = tokenized[sentence_index - 1] if sentence_index else []
            following = tokenized[sentence_index + 1] if sentence_index + 1 < len(tokenized) else []
            for start in range(0, len(target_ids), chunk_size):
                target = target_ids[start:start + chunk_size]
                budget = self.config["max_length"] - len(target) - self._pair_special_tokens
                context = _bounded_context(previous, following, budget, self.tokenizer.sep_token_id)
                encoded = self.tokenizer.prepare_for_model(
                    target, pair_ids=context, add_special_tokens=True,
                    padding=False, truncation=False, return_attention_mask=True,
                    return_token_type_ids=use_token_types,
                )
                # Special-token masks are ambiguous when text itself contains a
                # literal [SEP]/[CLS]. Build the same pair template with negative
                # sentinels, which can never be vocabulary or special-token IDs.
                template = self.tokenizer.build_inputs_with_special_tokens(
                    [-1] * len(target), [-2] * len(context),
                )
                target_mask = [token == -1 for token in template]
                if (len(template) != len(encoded["input_ids"])
                        or len(template) > self.config["max_length"]
                        or sum(target_mask) != len(target)):
                    raise ValueError("Tokenizer pair template did not preserve the target")
                example = {
                    "input_ids": list(encoded["input_ids"]),
                    "attention_mask": list(encoded["attention_mask"]),
                    "target_mask": target_mask,
                    "sentence_index": sentence_index,
                    "chunk_weight": len(target) / len(target_ids),
                }
                if use_token_types:
                    example["token_type_ids"] = list(encoded["token_type_ids"])
                if labels is not None:
                    example["label"] = role_ids[labels[sentence_index]]
                examples.append(example)
        return examples

    def collate(self, examples):
        """Dynamically right-pad a list of chunks, retaining training metadata."""
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        width = max(len(example["input_ids"]) for example in examples)
        batch = {}
        for key, padding, dtype in (
            ("input_ids", self.tokenizer.pad_token_id, torch.long),
            ("attention_mask", 0, torch.long),
            ("target_mask", False, torch.bool),
            ("token_type_ids", 0, torch.long),
        ):
            present = [key in example for example in examples]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"Inconsistent {key} fields in a batch")
            rows = []
            for example in examples:
                values = example[key]
                if len(values) != len(example["input_ids"]):
                    raise ValueError("Token fields must have aligned lengths")
                rows.append(list(values) + [padding] * (width - len(values)))
            batch[key] = torch.tensor(rows, dtype=dtype)
        batch["sentence_index"] = torch.tensor([row["sentence_index"] for row in examples], dtype=torch.long)
        batch["chunk_weight"] = torch.tensor([row["chunk_weight"] for row in examples], dtype=torch.float32)
        labelled = ["label" in row for row in examples]
        if any(labelled):
            if not all(labelled):
                raise ValueError("Cannot mix labelled and unlabelled chunks")
            batch["labels"] = torch.tensor([row["label"] for row in examples], dtype=torch.long)
        return batch

    def forward(self, batch=None, **kwargs):
        """Return chunk logits; only target tokens contribute to mean pooling."""
        if batch is None:
            batch = kwargs
        elif not isinstance(batch, Mapping) or kwargs:
            raise ValueError("Pass a batch mapping or keyword tensors")
        target_mask = batch["target_mask"].bool()
        attention_mask = batch["attention_mask"]
        if (target_mask.shape != batch["input_ids"].shape
                or target_mask.shape != attention_mask.shape
                or torch.any(target_mask & ~attention_mask.bool())
                or torch.any(target_mask.sum(dim=1) == 0)):
            raise ValueError("Each chunk needs an aligned, nonempty target mask on real tokens")
        inputs = {key: batch[key] for key in ("input_ids", "attention_mask", "token_type_ids") if key in batch}
        hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        weights = target_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        return self.to_roles(self.dropout(pooled))

    def aggregate_logits(self, examples, logits, sentence_count=None):
        """Sum chunk logits using their fractions of the original target tokens."""
        if logits.ndim != 2 or logits.shape != (len(examples), len(self.roles)):
            raise ValueError("Expected one seven-role logit row per chunk")
        if sentence_count is None:
            sentence_count = max((row["sentence_index"] for row in examples), default=-1) + 1
        if sentence_count < 0:
            raise ValueError("sentence_count must be nonnegative")
        combined = logits.new_zeros((sentence_count, len(self.roles)))
        if not examples:
            return combined
        indices = torch.tensor([row["sentence_index"] for row in examples], device=logits.device)
        weights = logits.new_tensor([row["chunk_weight"] for row in examples])
        if (torch.any(indices < 0) or torch.any(indices >= sentence_count)
                or not torch.isfinite(weights).all() or torch.any(weights <= 0)):
            raise ValueError("Invalid sentence indices or chunk weights")
        combined.index_add_(0, indices, logits * weights[:, None])
        return combined

    @torch.inference_mode()
    def predict(self, sentences, batch_size=None):
        """Return one canonical role per sentence in a single case."""
        sentences = list(sentences)
        if not sentences:
            return []
        batch_size = self.config["batch_size"] if batch_size is None else batch_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        examples = self.make_examples(sentences)
        device = self.to_roles.weight.device
        was_training = self.training
        self.eval()
        try:
            logits = []
            for start in range(0, len(examples), batch_size):
                batch = self.collate(examples[start:start + batch_size])
                batch = {key: value.to(device) for key, value in batch.items()}
                logits.append(self(batch).float().cpu())
            combined = self.aggregate_logits(examples, torch.cat(logits), len(sentences))
            return [self.roles[index] for index in combined.argmax(dim=1).tolist()]
        finally:
            self.train(was_training)

    def save(self, checkpoint, metadata=None):
        """Atomically save full fine-tuned weights and primitive configuration."""
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "roles": list(self.roles),
            "encoder": self.encoder,
            "encoder_model": self.encoder_model,
            "encoder_revision": self.encoder_revision,
            "config": dict(self.config),
            "backbone_config": self.backbone.config.to_dict(),
            "state": {name: value.detach().cpu() for name, value in self.state_dict().items()},
            "metadata": self.metadata if metadata is None else metadata,
        }
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_transformer_labeller(checkpoint, cache_dir=None, *, tokenizer=None, backbone=None):
    """Restore on CPU without downloading base weights or consuming torch RNG.

    Only the tokenizer is reconstructed from the pinned base. Its files may be
    supplied through the Hugging Face cache, or injected for offline unit tests.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported transformer labeller checkpoint format")
    encoder = payload.get("encoder")
    if (encoder not in MODEL_IDS or payload.get("encoder_model") != MODEL_IDS[encoder]
            or payload.get("encoder_revision") != MODEL_REVISIONS[encoder]):
        raise ValueError("Checkpoint transformer encoder differs from the pinned encoder")
    if tuple(payload.get("roles", ())) != ROLES:
        raise ValueError("Checkpoint must use the canonical rhetorical role order")
    backbone_config = dict(payload["backbone_config"])
    model_type = backbone_config.pop("model_type")
    if model_type != _MODEL_TYPES[encoder]:
        raise ValueError("Checkpoint backbone architecture differs from its encoder")
    with torch.random.fork_rng(devices=[]):
        if backbone is None:
            backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **backbone_config))
        model = TransformerRoleLabeller(
            encoder, **payload["config"], cache_dir=cache_dir,
            tokenizer=tokenizer, backbone=backbone,
        )
        model.load_state_dict(payload["state"])
    model.metadata = payload.get("metadata", {})
    model.eval()
    return model
