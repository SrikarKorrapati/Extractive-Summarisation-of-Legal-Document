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
