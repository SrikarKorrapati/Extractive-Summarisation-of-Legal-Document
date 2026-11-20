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


def test_entire_sentence_and_rare_number_shapes(tmp_path):
    from legal_lexrank.labeller import _encode

    long_sentence = "article 21 " + "word " * 110 + "appeal allowed"
    documents = [([long_sentence, "article 21 referred to 1859 and 456", long_sentence],
                  ["fact", "fact", "decision"])]
    checkpoint = tmp_path / "model.pt"
    model = fit_labeller(documents, ["fact", "decision"], 11, checkpoint, epochs=1)
    assert load_labeller(checkpoint).version == 2
    token_ids, lengths = _encode([long_sentence, "article 21 2024 77"], model.vocabulary, model.version)
    assert lengths.tolist() == [114, 4]
    assert token_ids[0, 113] == model.vocabulary["allowed"]
    assert token_ids[1, :4].tolist() == [
        model.vocabulary[word] for word in ("article", "21", "<four_digit_number>", "<number>")
    ]
    assert "1859" not in model.vocabulary and "2024" not in model.vocabulary
    with torch.no_grad():
        # The same long prefix followed by another ending must reach the word encoder.
        other = long_sentence.replace("allowed", "referred")
        original_emission = model(*_encode([long_sentence], model.vocabulary, model.version))
        other_emission = model(*_encode([other], model.vocabulary, model.version))
    assert not torch.allclose(original_emission, other_emission)


def test_version_one_checkpoint_retains_original_tokenization_and_rng(tmp_path):
    from legal_lexrank.labeller import RoleLabeller, _encode

    vocabulary = {"word": 2, "allowed": 3}
    old = RoleLabeller(vocabulary, ["fact", "decision"], version=1)
    checkpoint = tmp_path / "old.pt"
    torch.save({"version": 1, "vocabulary": vocabulary, "roles": list(old.roles),
                "state": old.state_dict()}, checkpoint)
    rng = torch.get_rng_state()
    restored = load_labeller(checkpoint)
    assert torch.equal(torch.get_rng_state(), rng)
    ids, lengths = _encode(["word " * 96 + "allowed", "2024"], vocabulary, restored.version)
    assert lengths.tolist() == [96, 1]
    assert ids[0].tolist() == [2] * 96
    assert ids[1, 0] == 1
