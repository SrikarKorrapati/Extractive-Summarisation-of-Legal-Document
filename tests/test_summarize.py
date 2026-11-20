"""Exercise the summary command using an isolated fitted-run fixture."""

import json

import pytest

from legal_lexrank import summarize


@pytest.mark.parametrize(("method", "expected"), [(None, "appeal statute"), ("unweighted", "appeal")])
def test_summary_uses_selected_weights_unless_overridden(tmp_path, capsys, monkeypatch, method, expected):
    sentences = ["appeal", "appeal statute", "appeal trial"]
    judgment = tmp_path / "judgment.txt"
    judgment.write_text("\n".join(sentences))
    results = tmp_path / "results"
    results.mkdir()
    (results / "tfidf.json").write_text(json.dumps({"selected_method": "mean"}))
    (results / "weights.json").write_text(json.dumps({"test": {"mean": {"Facts": 0.0, "Argument": 1.0}}}))

    class RoleLabeller:
        def predict(self, source):
            assert source == sentences
            return ["Facts", "Argument", "Argument"]

    def load_fixture(path):
        assert method != "unweighted", "unweighted LexRank requires no fitted artifacts"
        assert path == results / "role_labeller.pt"
        return RoleLabeller()

    monkeypatch.setattr(summarize, "load_labeller", load_fixture)
    args = [str(judgment), "--backend", "tfidf", "--results", str(results),
            "--cache", str(tmp_path / "cache"), "--ratio", "0.3"]
    if method:
        args.extend(["--method", method])
    summarize.main(args)
    assert capsys.readouterr().out.splitlines() == [expected]
