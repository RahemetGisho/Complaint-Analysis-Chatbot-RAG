import unittest
import numpy as np
import pandas as pd
import shutil
import os
import tempfile
import chromadb

from src.vector_store import build_vector_store


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        # ✅ Safe temporary DB (avoids Windows locking issues)
        self.test_dir = tempfile.mkdtemp()

        # small fake chunk dataframe
        self.chunk_df = pd.DataFrame({
            "chunk_id": ["1_0", "2_0", "2_1"],
            "chunk_text": [
                "I lost my credit card",
                "My account was hacked",
                "Unauthorized transaction detected"
            ],
            "complaint_id": [1, 2, 2],
            "product_category": [
                "Credit Card",
                "Savings Account",
                "Credit Card"
            ]
        })

        # fake embeddings (3 vectors, 384 dims)
        self.embeddings = np.random.rand(3, 384)

    # -----------------------------
    # 1. Collection is created
    # -----------------------------
    def test_collection_created(self):
        collection = build_vector_store(
            self.chunk_df,
            self.embeddings,
            persist_dir=self.test_dir,
            collection_name="test_chunks"
        )

        self.assertIsNotNone(collection)
        self.assertEqual(collection.name, "test_chunks")

    # -----------------------------
    # 2. Correct number of vectors stored
    # -----------------------------
    def test_vector_count(self):
        build_vector_store(
            self.chunk_df,
            self.embeddings,
            persist_dir=self.test_dir,
            collection_name="test_chunks"
        )

        client = chromadb.PersistentClient(path=self.test_dir)
        collection = client.get_collection("test_chunks")

        self.assertEqual(collection.count(), len(self.chunk_df))

    # -----------------------------
    # 3. Metadata correctness
    # -----------------------------
    def test_metadata_structure(self):
        build_vector_store(
            self.chunk_df,
            self.embeddings,
            persist_dir=self.test_dir,
            collection_name="test_chunks"
        )

        client = chromadb.PersistentClient(path=self.test_dir)
        collection = client.get_collection("test_chunks")

        data = collection.peek(limit=3)

        for meta in data["metadatas"]:
            self.assertIn("complaint_id", meta)
            self.assertIn("product_category", meta)

    # -----------------------------
    # 4. Documents stored correctly
    # -----------------------------
    def test_documents_stored(self):
        build_vector_store(
            self.chunk_df,
            self.embeddings,
            persist_dir=self.test_dir,
            collection_name="test_chunks"
        )

        client = chromadb.PersistentClient(path=self.test_dir)
        collection = client.get_collection("test_chunks")

        data = collection.peek(limit=3)

        self.assertEqual(len(data["documents"]), 3)

    # -----------------------------
    # 5. Embedding shape preserved
    # -----------------------------
    def test_embedding_dimension(self):
        build_vector_store(
            self.chunk_df,
            self.embeddings,
            persist_dir=self.test_dir,
            collection_name="test_chunks"
        )

        client = chromadb.PersistentClient(path=self.test_dir)
        collection = client.get_collection("test_chunks")

        data = collection.peek(limit=1)

        self.assertEqual(len(data["embeddings"][0]), 384)

    # -----------------------------
    # Cleanup after tests (FIXED)
    # -----------------------------
    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()