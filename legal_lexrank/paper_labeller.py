"""Pretrained sentence-vector variant of the paper's document BiLSTM-CRF.

The fixed legal sent2vec encoder supplies raw 200-dimensional sentence vectors.
Only the document BiLSTM, emission layer, and CRF are fitted. There is no
projection, input normalization, position feature, or auxiliary loss.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .data import ROLES
from .labeller import LinearChainCRF


CHECKPOINT_FORMAT = "paper-role-v1"
INPUT_SIZE = 200
DEFAULT_CACHE_DIR = Path(".cache/legal_lexrank")


def _checked_identity(identity=None):
    # Static identity: training and checkpoint loading need no native binary.
    from .legal_sent2vec import encoder_identity

    expected = encoder_identity()
    if identity is not None and identity != expected:
        raise ValueError("Checkpoint sentence encoder identity differs from legal sent2vec")
    if expected.get("dimensions") != INPUT_SIZE or expected.get("normalization") != "none":
        raise ValueError("The paper labeller requires raw 200-dimensional sentence vectors")
    return deepcopy(expected)


def _plain_metadata(metadata):
    """Keep checkpoint metadata compatible with the restricted tensor loader."""
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint metadata must contain finite JSON values") from error


class PaperRoleLabeller(nn.Module):
    """Raw sent2vec → document BiLSTM → dropout → emissions → linear-chain CRF."""

    def __init__(self, roles=ROLES, encoder_identity=None, *, hidden_size=100,
                 dropout=0.5, cache_dir=DEFAULT_CACHE_DIR, model_path=None):
        super().__init__()
        self.roles = tuple(roles)
        if self.roles != tuple(ROLES):
            raise ValueError("Paper labeller roles must use the seven canonical roles in order")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Hidden size must be a positive integer")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Dropout must be finite and in [0, 1)")
        self.input_size = INPUT_SIZE
        self.encoder_identity = _checked_identity(encoder_identity)
        self.config = {"hidden_size": hidden_size, "dropout": float(dropout)}
        self.cache_dir = Path(cache_dir)
        self.model_path = None if model_path is None else Path(model_path)
        self.metadata = {}
        self.document_lstm = nn.LSTM(
            INPUT_SIZE, hidden_size, bidirectional=True, batch_first=True,
            device="cpu", dtype=torch.float32,
        )
        self.dropout = nn.Dropout(dropout)
        self.to_roles = nn.Linear(2 * hidden_size, len(self.roles), device="cpu", dtype=torch.float32)
        self.crf = LinearChainCRF(len(self.roles)).to(device="cpu", dtype=torch.float32)

    def _vectors(self, vectors):
        parameter = self.to_roles.weight
        vectors = torch.as_tensor(vectors, dtype=parameter.dtype, device=parameter.device)
        if vectors.ndim != 2 or vectors.shape[1] != INPUT_SIZE:
            raise ValueError("Expected sentence vectors with shape (sentences, 200)")
        if not torch.isfinite(vectors).all():
            raise ValueError("Sentence vectors must be finite")
        return vectors

    def _labels(self, labels, count, device):
        labels = torch.as_tensor(labels, device=device)
        if labels.ndim != 1 or len(labels) != count or len(labels) == 0:
            raise ValueError("Training needs one integer role label per sentence")
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Training labels must be integer role indices")
        labels = labels.to(dtype=torch.long)
        if (labels < 0).any() or (labels >= len(self.roles)).any():
            raise ValueError("Training labels must index the configured roles")
        return labels

    def forward(self, vectors):
        """Return one emission vector per sentence without changing input scale."""
        vectors = self._vectors(vectors)
        if len(vectors) == 0:
            return vectors.new_empty((0, len(self.roles)))
        contextual, _ = self.document_lstm(vectors.unsqueeze(0))
        return self.to_roles(self.dropout(contextual.squeeze(0)))

    def loss(self, vectors, labels):
        """Full-sequence CRF negative log likelihood, with no auxiliary objective."""
        emissions = self(vectors)
        labels = self._labels(labels, len(emissions), emissions.device)
        return self.crf.neg_log_likelihood(emissions, labels)

    def loss_batch(self, vectors, labels):
        """Sum independent document CRF losses with packed BiLSTM computation.

        Packing prevents padding from entering the backward LSTM context. The
        CRF recurrence freezes each document at its final real sentence and
        uses the same start, transition, and end parameters as ``loss``.
        """
        vectors, labels = list(vectors), list(labels)
        if not vectors or len(vectors) != len(labels):
            raise ValueError("A training batch needs aligned nonempty document lists")
        vectors = [self._vectors(values) for values in vectors]
        labels = [self._labels(gold, len(values), values.device)
                  for values, gold in zip(vectors, labels)]
        lengths = torch.tensor([len(values) for values in vectors], dtype=torch.long)
        packed = pack_padded_sequence(
            pad_sequence(vectors, batch_first=True), lengths,
            batch_first=True, enforce_sorted=False,
        )
        contextual, _ = self.document_lstm(packed)
        contextual, _ = pad_packed_sequence(contextual, batch_first=True)
        emissions = self.to_roles(self.dropout(contextual))
        gold = pad_sequence(labels, batch_first=True)
        lengths = lengths.to(device=emissions.device)
        valid = torch.arange(emissions.shape[1], device=emissions.device)[None, :] < lengths[:, None]

        scores = self.crf.start + emissions[:, 0]
        for position in range(1, emissions.shape[1]):
            following = torch.logsumexp(
                scores[:, :, None] + self.crf.transitions[None, :, :], dim=1,
            ) + emissions[:, position]
            scores = torch.where(valid[:, position, None], following, scores)
        partition = torch.logsumexp(scores + self.crf.end, dim=1)
        final_labels = gold.gather(1, (lengths - 1)[:, None]).squeeze(1)
        gold_scores = self.crf.start[gold[:, 0]] + self.crf.end[final_labels]
        gold_scores = gold_scores + (emissions.gather(2, gold[:, :, None]).squeeze(2) * valid).sum(dim=1)
        gold_scores = gold_scores + (
            self.crf.transitions[gold[:, :-1], gold[:, 1:]] * valid[:, 1:]
        ).sum(dim=1)
        return (partition - gold_scores).sum()

    @torch.inference_mode()
    def predict_vectors(self, vectors):
        """Decode deterministically and restore every module's training mode."""
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            emissions = self(vectors)
            if len(emissions) == 0:
                return []
            return [self.roles[index] for index in self.crf.decode(emissions)]
        finally:
            for module, training in modes:
                module.training = training

    def predict(self, sentences):
        """Embed a new judgment with the same fixed legal sent2vec encoder."""
        sentences = list(sentences)
        if not sentences:
            return []
        from .legal_sent2vec import sentence_vectors

        _checked_identity(self.encoder_identity)
        vectors = sentence_vectors(sentences, cache_dir=self.cache_dir, model_path=self.model_path)
        return self.predict_vectors(vectors)

    def save(self, checkpoint, metadata=None):
        """Atomically save portable tensors and verified encoder metadata."""
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "input_size": INPUT_SIZE,
            "roles": list(self.roles),
            "encoder_identity": _checked_identity(self.encoder_identity),
            "config": dict(self.config),
            "state": {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()},
            "metadata": _plain_metadata(self.metadata if metadata is None else metadata),
        }
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_paper_labeller(checkpoint, cache_dir=None, model_path=None):
    """Load on CPU without native sent2vec or consuming the caller's RNG state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported paper labeller checkpoint format")
    if payload.get("input_size") != INPUT_SIZE:
        raise ValueError("Paper labeller checkpoint must use 200-dimensional sentence vectors")
    if not isinstance(payload.get("config"), dict) or set(payload["config"]) != {"hidden_size", "dropout"}:
        raise ValueError("Unsupported paper labeller configuration")
    if not isinstance(payload.get("encoder_identity"), dict):
        raise ValueError("Paper labeller checkpoint is missing its encoder identity")
    identity = _checked_identity(payload["encoder_identity"])
    metadata = _plain_metadata(payload.get("metadata", {}))
    with torch.random.fork_rng(devices=[]):
        model = PaperRoleLabeller(
            payload["roles"], identity, **payload["config"],
            cache_dir=DEFAULT_CACHE_DIR if cache_dir is None else cache_dir,
            model_path=model_path,
        )
    model.load_state_dict(payload["state"], strict=True)
    model.metadata = metadata
    model.eval()
    return model
