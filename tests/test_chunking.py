import unittest
import pandas as pd

from src.chunking import chunk_dataframe

class TestChunking(unittest.TestCase):

    def setUp(self):
        # small fake dataset
        self.df = pd.DataFrame({
            "Complaint ID": [1001, 1002],
            "cleaned_narrative": [
                "This is a long complaint text. " * 20,
                "Another complaint about banking issue. " * 15
            ],
            "product_category": [
                "Credit Card",
                "Savings Account"
            ]
        })

    # -----------------------------
    # 1. Output is DataFrame
    # -----------------------------
    def test_output_type(self):
        result = chunk_dataframe(self.df)

        self.assertIsInstance(result, pd.DataFrame)

    # -----------------------------
    # 2. Required columns exist
    # -----------------------------
    def test_required_columns(self):
        result = chunk_dataframe(self.df)

        expected_cols = {
            "chunk_id",
            "complaint_id",
            "chunk_index",
            "chunk_text",
            "total_chunks",
            "product_category"
        }

        self.assertTrue(expected_cols.issubset(set(result.columns)))

    # -----------------------------
    # 3. Each complaint produces chunks
    # -----------------------------
    def test_chunk_expansion(self):
        result = chunk_dataframe(self.df, chunk_size=50, chunk_overlap=10)

        # should expand (more rows than input)
        self.assertGreater(len(result), len(self.df))

    # -----------------------------
    # 4. Chunk ID format check
    # -----------------------------
    def test_chunk_id_format(self):
        result = chunk_dataframe(self.df, chunk_size=50, chunk_overlap=10)

        for cid in result["chunk_id"]:
            self.assertRegex(cid, r"^\d+_\d+$")  # e.g. 1001_0

    # -----------------------------
    # 5. Chunk index correctness
    # -----------------------------
    def test_chunk_index_sequence(self):
        result = chunk_dataframe(self.df, chunk_size=50, chunk_overlap=10)

        grouped = result.groupby("complaint_id")["chunk_index"].apply(list)

        for indices in grouped:
            self.assertEqual(indices, list(range(len(indices))))

    # -----------------------------
    # 6. Total chunks consistency
    # -----------------------------
    def test_total_chunks_consistency(self):
        result = chunk_dataframe(self.df, chunk_size=50, chunk_overlap=10)

        grouped = result.groupby("complaint_id")

        for _, group in grouped:
            expected_total = group["total_chunks"].iloc[0]
            self.assertTrue((group["total_chunks"] == expected_total).all())


if __name__ == "__main__":
    unittest.main()