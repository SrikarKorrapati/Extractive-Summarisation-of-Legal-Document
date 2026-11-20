import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from legal_lexrank import encoders


class EncoderTests(unittest.TestCase):
    def test_tfidf_is_per_document_and_cache_preserves_sentence_order(self):
        with tempfile.TemporaryDirectory() as cache:
            sentences = ["court appeal", "unrelated", "court appeal"]
            expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            matrix = encoders.similarity_matrix("tfidf", sentences, Path(cache))
            np.testing.assert_allclose(matrix, expected, atol=1e-6)
            with patch.object(encoders, "_tfidf", side_effect=AssertionError("cache miss")):
                reloaded = encoders.similarity_matrix("tfidf", sentences, Path(cache))
                np.testing.assert_array_equal(matrix, reloaded)
            reverse = encoders.similarity_matrix("tfidf", [sentences[i] for i in (0, 2, 1)], Path(cache))
            self.assertAlmostEqual(reverse[0, 1], 1, places=5)
            self.assertEqual(reverse[0, 2], 0)

    def test_empty_and_tokenless_text(self):
        with tempfile.TemporaryDirectory() as cache:
            matrix = encoders.similarity_matrix("tfidf", ["...", "??"], Path(cache))
            np.testing.assert_array_equal(matrix, np.zeros((2, 2)))
            self.assertEqual(encoders.similarity_matrix("sbert", [], Path(cache)).shape, (0, 0))
            with self.assertRaises(ValueError):
                encoders.similarity_matrix("tfidf", [""], Path(cache))

    def test_openai_chunked_batches_reuse_results_and_restore_api_order(self):
        class Client:
            def __init__(self, **kwargs):
                self.embeddings = self

            def create(self, *, model, input, dimensions, encoding_format):
                calls.append(input)
                vectors = []
                for i, ids in enumerate(input):
                    axis = ids[0] - 1
                    embedding = [0.] * dimensions
                    embedding[axis] = 1.
                    vectors.append(SimpleNamespace(index=i, embedding=embedding))
                return SimpleNamespace(data=list(reversed(vectors)))

            def close(self):
                pass

        calls = []
        codec = SimpleNamespace(encode_ordinary=lambda s: [int(i) for i in s])
        with (tempfile.TemporaryDirectory() as cache, patch.dict("os.environ", OPENAI_API_KEY="unused"),
              patch("openai.OpenAI", Client), patch("tiktoken.encoding_for_model", return_value=codec),
              patch.object(encoders, "OPENAI_CONTEXT", 2), patch.object(encoders, "OPENAI_BATCH_TOKENS", 3)):
            matrix = encoders.similarity_matrix("openai", ["112", "2", "1"], Path(cache))
            self.assertEqual(matrix.shape, (3, 3))
            self.assertAlmostEqual(float(matrix[0, 1]), 1 / np.sqrt(5), places=5)
            self.assertAlmostEqual(float(matrix[0, 2]), 2 / np.sqrt(5), places=5)
            self.assertTrue(all(sum(map(len, batch)) <= 3 for batch in calls))
            self.assertEqual(sum(map(len, calls)), 3)
            before = len(calls)
            encoders.similarity_matrix("openai", ["2", "112"], Path(cache))
            self.assertEqual(len(calls), before)


def _overlap_worker(cache, sentences, barrier):
    """Start two independent clients at the same time against overlapping token IDs."""
    import time

    class Client:
        def __init__(self, **kwargs):
            self.embeddings = self

        def create(self, *, model, input, dimensions, encoding_format):
            with (cache / "requests.log").open("a") as log:
                log.writelines(f"{ids[0]}\n" for ids in input)
            # Let the other process reach its cache lookup while this call is in flight.
            time.sleep(0.2)
            return SimpleNamespace(data=[
                SimpleNamespace(index=i, embedding=[float(ids[0]), 1.]) for i, ids in enumerate(input)
            ])

        def close(self):
            pass

    codec = SimpleNamespace(encode_ordinary=lambda s: [int(s)])
    with (patch.dict("os.environ", OPENAI_API_KEY="unused"), patch("openai.OpenAI", Client),
          patch("tiktoken.encoding_for_model", return_value=codec), patch.object(encoders, "OPENAI_DIMENSIONS", 2)):
        barrier.wait(timeout=20)
        encoders.similarity_matrix("openai", sentences, cache)


def test_concurrent_processes_pay_once_for_overlapping_sentences(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    gate = ctx.Barrier(2)
    workers = [ctx.Process(target=_overlap_worker, args=(tmp_path, pair, gate))
               for pair in (["1", "2"], ["2", "3"])]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    assert sorted((tmp_path / "requests.log").read_text().splitlines()) == ["1", "2", "3"]


def test_encoder_settings_invalidate_a_cached_matrix(tmp_path):
    sentences = ["A legal judgment", "The appeal succeeds"]
    with patch.object(encoders, "_local_embeddings", side_effect=[
        np.eye(2, dtype=np.float32), np.ones((2, 2), dtype=np.float32)
    ]) as embed:
        first = encoders.similarity_matrix("sbert", sentences, tmp_path)
        with patch.dict(encoders.MAX_LENGTH, sbert=32):
            second = encoders.similarity_matrix("sbert", sentences, tmp_path)
            encoders.similarity_matrix("sbert", sentences, tmp_path)
        assert embed.call_count == 2
        assert first[0, 1] == 0
        np.testing.assert_allclose(second[0, 1], 1, atol=1e-6)
