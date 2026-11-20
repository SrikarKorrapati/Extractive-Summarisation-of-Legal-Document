"""Exercise fixed role weights and the similarity-only summary command."""

import numpy as np
import pytest

from legal_lexrank import summarize


@pytest.mark.parametrize(("unweighted", "expected"), [(False, "court allows appeal"), (True, "appeal")])
def test_summary_uses_fixed_defaults_with_only_a_checkpoint(tmp_path, capsys, monkeypatch, unweighted, expected):
    sentences = ["appeal", "court allows appeal", "earlier decision"]
    judgment = tmp_path / "judgment.txt"
    judgment.write_text("\n".join(sentences))
    results = tmp_path / "results"

    class RoleLabeller:
        def predict(self, source):
            assert source == sentences
            return ["Facts", "Ruling by Present Court", "Precedent"]

    def load_fixture(path, cache_dir=None):
        assert not unweighted
        assert path == results / "role_labeller.pt"
        assert cache_dir == tmp_path / "cache" / "similarities"
        return RoleLabeller()

    def equally_similar(backend, source, cache_dir):
        assert source == sentences
        return np.ones((3, 3))

    monkeypatch.setattr(summarize, "load_labeller", load_fixture)
    monkeypatch.setattr(summarize, "similarity_matrix", equally_similar)
    args = [str(judgment), "--backend", "tfidf", "--results", str(results),
            "--cache", str(tmp_path / "cache"), "--ratio", "0.3"]
    if unweighted:
        args.append("--unweighted")
    summarize.main(args)
    assert capsys.readouterr().out.splitlines() == [expected]
