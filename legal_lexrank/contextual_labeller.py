"""Rhetorical-role tagging with pretrained sentence vectors and a document BiLSTM-CRF."""

import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from . import encoders
from .labeller import LinearChainCRF


CHECKPOINT_FORMAT = "contextual-role-v1"


class ContextualRoleLabeller(nn.Module):
    """Read a judgment's sentence vectors in order and decode its rhetorical roles.

    The caller supplies normalized pretrained vectors. Optional relative sentence
    positions let the model learn where roles tend to occur within a judgment.
    """

    def __init__(self, input_size, roles, encoder, *, hidden_size=128, dropout=0.3,
                 use_position=True, center_inputs=False, cache_dir=Path(".cache/legal_lexrank")):
        super().__init__()
        self.roles = tuple(roles)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("Roles must be nonempty and unique")
        if encoder not in ("openai", "sbert", "legalbert"):
            raise ValueError("A contextual labeller requires a pretrained sentence encoder")
        if input_size < 1 or hidden_size < 1:
            raise ValueError("Embedding and hidden dimensions must be positive")
        self.input_size = input_size
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.config = {"hidden_size": hidden_size, "dropout": dropout, "use_position": use_position}
        if center_inputs:
            self.config["center_inputs"] = True
            self.register_buffer("feature_mean", torch.zeros(input_size))
        self.metadata = {}
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.LayerNorm(2 * hidden_size),
            nn.GELU(),
        )
        self.document_lstm = nn.LSTM(
            2 * hidden_size + 2 * int(use_position), hidden_size,
            bidirectional=True, batch_first=True,
        )
        self.to_roles = nn.Linear(2 * hidden_size, len(self.roles))
        self.crf = LinearChainCRF(len(self.roles))

    def forward(self, vectors):
        parameter = self.to_roles.weight
        vectors = torch.as_tensor(vectors, dtype=parameter.dtype, device=parameter.device)
        if vectors.ndim != 2 or vectors.shape[1] != self.input_size:
            raise ValueError(f"Expected sentence vectors with shape (sentences, {self.input_size})")
        if not torch.isfinite(vectors).all():
            raise ValueError("Sentence vectors must be finite")
        count = len(vectors)
        if count == 0:
            return vectors.new_empty((0, len(self.roles)))
        if self.config.get("center_inputs", False):
            vectors = F.normalize(vectors - self.feature_mean, dim=1, eps=1e-12)
        vectors = self.projection(vectors)
        if self.config["use_position"]:
            position = torch.linspace(0, 1, count, dtype=vectors.dtype, device=vectors.device)
            vectors = torch.cat((vectors, position[:, None], 1 - position[:, None]), dim=1)
        contextual, _ = self.document_lstm(self.dropout(vectors).unsqueeze(0))
        return self.to_roles(self.dropout(contextual.squeeze(0)))

    def loss(self, vectors, labels, class_weights=None, auxiliary_strength=0.3):
        """Mean sequence loss plus optionally class-weighted sentence supervision."""
        emissions = self(vectors)
        labels = torch.as_tensor(labels, dtype=torch.long, device=emissions.device)
        if labels.ndim != 1 or len(labels) != len(emissions) or len(labels) == 0:
            raise ValueError("Training needs one integer role label per sentence")
        if (labels < 0).any() or (labels >= len(self.roles)).any():
            raise ValueError("Training labels must index the configured roles")
        if auxiliary_strength < 0:
            raise ValueError("Auxiliary loss strength must be nonnegative")
        loss = self.crf.neg_log_likelihood(emissions, labels) / len(labels)
        if auxiliary_strength == 0:
            return loss
        weights = None
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=emissions.dtype, device=emissions.device)
            if (weights.shape != (len(self.roles),) or not torch.isfinite(weights).all()
                    or (weights <= 0).any()):
                raise ValueError("Class weights must contain one finite positive value per role")
        return loss + auxiliary_strength * F.cross_entropy(emissions, labels, weight=weights)

    @torch.inference_mode()
    def predict_vectors(self, vectors):
        if len(vectors) == 0:
            return []
        was_training = self.training
        self.eval()
        try:
            indices = self.crf.decode(self(vectors))
            return [self.roles[index] for index in indices]
        finally:
            self.train(was_training)

    def predict(self, sentences):
        if not sentences:
            return []
        vectors = encoders.sentence_embeddings(self.encoder, sentences, self.cache_dir)
        return self.predict_vectors(vectors)

    def save(self, checkpoint, metadata=None):
        """Atomically write portable tensors and primitive configuration metadata."""
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "input_size": self.input_size,
            "roles": list(self.roles),
            "encoder": self.encoder,
            "encoder_model": encoders.MODEL_IDS[self.encoder],
            "encoder_revision": encoders.MODEL_REVISIONS.get(self.encoder),
            "config": self.config,
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


def load_contextual_labeller(checkpoint, cache_dir=None):
    """Load a saved contextual model on CPU without consuming caller RNG state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported contextual labeller checkpoint format")
    encoder = payload["encoder"]
    if (payload["encoder_model"] != encoders.MODEL_IDS.get(encoder)
            or payload["encoder_revision"] != encoders.MODEL_REVISIONS.get(encoder)):
        raise ValueError("Checkpoint sentence encoder differs from the configured encoder")
    with torch.random.fork_rng(devices=[]):
        model = ContextualRoleLabeller(
            payload["input_size"], payload["roles"], encoder, **payload["config"],
            cache_dir=Path(".cache/legal_lexrank") if cache_dir is None else cache_dir,
        )
    model.load_state_dict(payload["state"])
    model.metadata = payload["metadata"]
    model.eval()
    return model
