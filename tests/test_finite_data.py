import unittest
from finite_data import learner


class FiniteInputTests(unittest.TestCase):
    def test_observable_schema_and_causal_prefix(self):
        prefix = learner(42, 1, steps=6)
        self.assertEqual(prefix, learner(42, 1, steps=9)[:6])
        self.assertEqual(set(prefix[0]), {"learner", "exercise", "concept", "correct", "timestamp"})
        self.assertTrue(all(a["timestamp"] < b["timestamp"] for a, b in zip(prefix, prefix[1:])))
        self.assertTrue(all(row["correct"] in (0, 1) for row in prefix))
        self.assertNotEqual(prefix, learner(42, 2, steps=6))
