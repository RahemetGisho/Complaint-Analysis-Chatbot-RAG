import unittest
import pandas as pd

from src.sampling import stratified_sample


class TestStratifiedSampling(unittest.TestCase):

    def setUp(self):
        # Create synthetic dataset with imbalance
        self.df = pd.DataFrame({
            "text": range(100),
            "product_category": (
                ["Credit Card"] * 50 +
                ["Savings Account"] * 30 +
                ["Money Transfer"] * 20
            )
        })

    # -----------------------------
    # 1. Output size correctness
    # -----------------------------
    def test_sample_size(self):
        target_n = 40
        sample = stratified_sample(self.df, target_n)

        self.assertEqual(len(sample), target_n)

    # -----------------------------
    # 2. All categories preserved
    # -----------------------------
    def test_all_categories_present(self):
        sample = stratified_sample(self.df, target_n=40)

        categories = set(sample["product_category"].unique())

        self.assertEqual(
            categories,
            {"Credit Card", "Savings Account", "Money Transfer"}
        )

    # -----------------------------
    # 3. No category exceeds available data
    # -----------------------------
    def test_not_over_sampling(self):
        sample = stratified_sample(self.df, target_n=200)

        counts = self.df["product_category"].value_counts()
        sample_counts = sample["product_category"].value_counts()

        for cat in sample_counts.index:
            self.assertLessEqual(sample_counts[cat], counts[cat])

    # -----------------------------
    # 4. Reproducibility test
    # -----------------------------
    def test_reproducibility(self):
        sample1 = stratified_sample(self.df, target_n=40, random_state=42)
        sample2 = stratified_sample(self.df, target_n=40, random_state=42)

        pd.testing.assert_frame_equal(sample1, sample2)

    # -----------------------------
    # 5. Proportionality sanity check
    # -----------------------------
    def test_proportional_distribution(self):
        sample = stratified_sample(self.df, target_n=60)

        dist = sample["product_category"].value_counts(normalize=True)
        original_dist = self.df["product_category"].value_counts(normalize=True)

        # allow small deviation due to rounding
        for cat in original_dist.index:
            self.assertAlmostEqual(
                dist[cat],
                original_dist[cat],
                delta=0.15
            )


if __name__ == "__main__":
    unittest.main()