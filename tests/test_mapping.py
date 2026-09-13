import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MappingTest(unittest.TestCase):
    def test_hf_and_weighted_sum(self):
        from scripts.convert_cd import mapped_vectors

        counts = torch.tensor([[0, 2], [3, 1]])
        indices = torch.tensor([[0, 1]])
        codewords = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        hf = mapped_vectors(counts, indices, codewords, "s2t-hf")
        ws = mapped_vectors(counts, indices, codewords, "s2t-ws")
        torch.testing.assert_close(hf, torch.tensor([[[0.0, 1.0], [2.0, 0.0]]]))
        torch.testing.assert_close(ws, torch.tensor([[[0.0, 0.75], [2.0, 0.5]]]))

    def test_invalid_dimensions_fail(self):
        from scripts.convert_cd import mapped_vectors

        with self.assertRaises(ValueError):
            mapped_vectors(torch.zeros(2, 3), torch.zeros(1, 2, dtype=torch.long),
                           torch.zeros(2, 80), "s2t-hf")


if __name__ == "__main__":
    unittest.main()
