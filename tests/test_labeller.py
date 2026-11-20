from itertools import product

import pytest
import torch

from legal_lexrank.labeller import LinearChainCRF, fit_labeller, load_labeller


@pytest.mark.parametrize("length", [1, 4])
def test_crf_matches_exhaustive_paths(length):
    crf = LinearChainCRF(3)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for param in crf.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    emissions = torch.randn(length, 3, generator=generator, requires_grad=True)
    paths = list(product(range(3), repeat=length))
    scores = torch.stack([
        crf.start[path[0]] + crf.end[path[-1]]
        + sum(emissions[i, state] for i, state in enumerate(path))
        + sum(crf.transitions[a, b] for a, b in zip(path, path[1:]))
        for path in paths
    ])
    loss = crf.neg_log_likelihood(emissions, torch.tensor(paths[0]))
    torch.testing.assert_close(loss, scores.logsumexp(dim=0) - scores[0])
    assert crf.decode(emissions) == list(paths[int(scores.argmax())])
    loss.backward()
    assert torch.isfinite(emissions.grad).all()


def test_fitted_role_tagger_roundtrip(tmp_path):
    documents = [(["facts here", "our ruling", "final ruling"], ["fact", "decision", "decision"])]
    checkpoint = tmp_path / "tagger.pt"
    model = fit_labeller(documents, ["fact", "decision"], 17, checkpoint, epochs=2)
    unseen = ["brandnewtoken here", "short", "short", ""]
    prediction = model.predict(unseen)
    assert len(prediction) == len(unseen)
    assert set(prediction) <= {"fact", "decision"}
    assert "brandnewtoken" not in model.vocabulary
    assert load_labeller(checkpoint).predict(unseen) == prediction
    assert model.predict([]) == []
