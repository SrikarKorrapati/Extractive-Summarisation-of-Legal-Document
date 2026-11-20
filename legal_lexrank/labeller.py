"""Supervised hierarchical BiLSTM-CRF for sentence-level rhetorical roles.

Each sentence is encoded by a word BiLSTM. A second BiLSTM sees the sentences
of one judgment in order. The CRF scores a role sequence, including its first
role, last role, and adjacent role transitions.
"""

from collections import Counter
from pathlib import Path
import os
import random
import re
import tempfile

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
_MAX_TOKENS = 96
_PAD, _UNKNOWN = 0, 1


def _tokens(sentence):
    return _TOKEN_PATTERN.findall(sentence.casefold())[:_MAX_TOKENS]


def _encode(sentences, vocabulary):
    # Short empty inputs use one unknown word so the packed LSTM stays defined.
    rows = [
        torch.tensor([vocabulary.get(word, _UNKNOWN) for word in _tokens(sentence)] or [_UNKNOWN])
        for sentence in sentences
    ]
    lengths = torch.tensor([len(row) for row in rows])
    return pad_sequence(rows, batch_first=True, padding_value=_PAD), lengths


class LinearChainCRF(nn.Module):
    """Unconstrained linear-chain CRF with transitions indexed [previous, next]."""

    def __init__(self, num_roles):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(num_roles))
        self.transitions = nn.Parameter(torch.zeros(num_roles, num_roles))
        self.end = nn.Parameter(torch.zeros(num_roles))

    def neg_log_likelihood(self, emissions, labels):
        forward = self.start + emissions[0]
        for row in emissions[1:]:
            forward = torch.logsumexp(forward[:, None] + self.transitions, dim=0) + row
        log_partition = torch.logsumexp(forward + self.end, dim=0)
        gold = self.start[labels[0]] + self.end[labels[-1]]
        gold = gold + emissions.gather(1, labels[:, None]).sum()
        gold = gold + self.transitions[labels[:-1], labels[1:]].sum()
        return log_partition - gold

    def decode(self, emissions):
        scores = self.start + emissions[0]
        backpointers = []
        for row in emissions[1:]:
            best, previous = (scores[:, None] + self.transitions).max(dim=0)
            backpointers.append(previous)
            scores = best + row
        best_last = int((scores + self.end).argmax())
        path = [best_last]
        for previous in reversed(backpointers):
            path.append(int(previous[path[-1]]))
        return list(reversed(path))


class RoleLabeller(nn.Module):
    def __init__(self, vocabulary, roles):
        super().__init__()
        self.vocabulary = vocabulary
        self.roles = tuple(roles)
        self.embedding = nn.Embedding(len(vocabulary) + 2, 48, padding_idx=_PAD)
        self.word_lstm = nn.LSTM(48, 24, bidirectional=True, batch_first=True)
        self.document_lstm = nn.LSTM(48, 32, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.to_roles = nn.Linear(64, len(roles))
        self.crf = LinearChainCRF(len(roles))

    def forward(self, token_ids, lengths):
        words = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(words, lengths, batch_first=True, enforce_sorted=False)
        _, (last_hidden, _) = self.word_lstm(packed)
        sentences = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        contextual, _ = self.document_lstm(self.dropout(sentences).unsqueeze(0))
        return self.to_roles(self.dropout(contextual.squeeze(0)))

    @torch.inference_mode()
    def predict(self, sentences):
        if not sentences:
            return []
        self.eval()
        emissions = self(*_encode(sentences, self.vocabulary))
        return [self.roles[i] for i in self.crf.decode(emissions)]


def fit_labeller(documents, roles, seed, checkpoint, *, epochs=8):
    """Fit on the supplied (sentences, labels) documents; atomically save weights.

    The CPU implementation learns all parameters from scratch. Vocabulary entries
    must appear at least twice in the supplied training cases. The word encoder
    sees the first 96 tokens; summaries still use the full source sentence.
    """
    documents = list(documents)
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Roles must be nonempty and unique")
    if not documents or epochs < 1:
        raise ValueError("Training requires documents and at least one epoch")
    role_ids = {role: i for i, role in enumerate(roles)}
    counts = Counter()
    for sentences, labels in documents:
        if not sentences or len(sentences) != len(labels):
            raise ValueError("Every training document needs aligned sentences and labels")
        if set(labels) - set(role_ids):
            raise ValueError("Training document contains an unknown role label")
        for sentence in sentences:
            counts.update(_tokens(sentence))
    vocabulary = {
        word: i + 2 for i, word in enumerate(sorted(w for w, count in counts.items() if count >= 2))
    }
    encoded = [
        (*_encode(sentences, vocabulary), torch.tensor([role_ids[label] for label in labels]))
        for sentences, labels in documents
    ]
    rng = random.Random(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = RoleLabeller(vocabulary, roles)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        order = list(range(len(encoded)))
        for _ in range(epochs):
            rng.shuffle(order)
            for i in order:
                token_ids, lengths, labels = encoded[i]
                optimizer.zero_grad(set_to_none=True)
                loss = model.crf.neg_log_likelihood(model(token_ids, lengths), labels) / len(labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        torch.save(
            {"version": 1, "vocabulary": vocabulary, "roles": list(roles), "state": model.state_dict(),
             "seed": seed, "epochs": epochs},
            temp_path,
        )
        os.replace(temp_path, checkpoint)
    finally:
        temp_path.unlink(missing_ok=True)
    model.eval()
    return model


def load_labeller(checkpoint):
    """Load a fitted role tagger for sentence-aligned Viterbi predictions."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["version"] != 1:
        raise ValueError("Unsupported labeller checkpoint version")
    model = RoleLabeller(state["vocabulary"], state["roles"])
    model.load_state_dict(state["state"])
    model.eval()
    return model
