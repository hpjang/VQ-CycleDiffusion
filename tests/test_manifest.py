import tempfile
import unittest
from pathlib import Path

from scripts.make_manifest import SPEAKERS, rows


class ManifestTest(unittest.TestCase):
    def test_grid_is_10440_with_distinct_sentence_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for speaker in SPEAKERS:
                folder = root / speaker
                folder.mkdir()
                for number in range(2, 32):
                    (folder / f"{speaker}_{number:03d}.wav").touch()
            generated = list(rows(root, root / "out"))
            self.assertEqual(len(generated), 12 * 30 * 29)
            for row in generated:
                source_id = Path(row["source_wav"]).stem.split("_")[1]
                reference_id = Path(row["reference_wav"]).stem.split("_")[1]
                self.assertNotEqual(source_id, reference_id)

    def test_missing_utterance_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for speaker in SPEAKERS:
                (root / speaker).mkdir()
            with self.assertRaisesRegex(ValueError, "Expected 30"):
                list(rows(root, root / "out"))


if __name__ == "__main__":
    unittest.main()
