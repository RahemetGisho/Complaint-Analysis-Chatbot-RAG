import unittest
import numpy as np

from src.embedding import load_model, embed_texts


class DummyModel:
    """Fake model to avoid downloading real transformer."""

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        # return fake embeddings (384-dim like MiniLM)
        return np.array([[0.1] * 384 for _ in texts])


class TestEmbedding(unittest.TestCase):

    # -----------------------------
    # 1. Model loading test
    # -----------------------------
    def test_load_model(self):
        model = load_model()

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "encode"))

    # -----------------------------
    # 2. Embedding output shape
    # -----------------------------
    def test_embed_shape(self):
        model = DummyModel()

        texts = [
            "I lost my credit card",
            "My account was hacked"
        ]

        embeddings = embed_texts(model, texts)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 384))

    # -----------------------------
    # 3. Embedding content type
    # -----------------------------
    def test_embed_values(self):
        model = DummyModel()

        texts = ["test text"]

        embeddings = embed_texts(model, texts)

        self.assertEqual(len(embeddings[0]), 384)
        self.assertAlmostEqual(float(embeddings[0][0]), 0.1)

    # -----------------------------
    # 4. Empty input handling
    # -----------------------------
    def test_empty_input(self):
        model = DummyModel()

        embeddings = embed_texts(model, [])

        self.assertEqual(len(embeddings), 0)


if __name__ == "__main__":
    unittest.main()