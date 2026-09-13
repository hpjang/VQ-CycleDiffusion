import tempfile
import unittest
from pathlib import Path

from scripts.setup_drive_assets import DATA_DIRS, setup


class SetupDriveAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.assets = root / "assets"
        self.repo = root / "repo"
        self.repo.mkdir()
        (self.repo / "params.py").touch()
        (self.repo / "hifi-gan").mkdir()
        for name in DATA_DIRS:
            (self.assets / name / "wavs").mkdir(parents=True)
        for filename in (
            "checkpts/spk_encoder/pretrained.pt",
            "log/codebook_stock_255_exclude/p236_exclude/codebook_stock_p236_4096.pt",
            "hifi-gan/generator_universal.pth",
            "vc_255.pt",
        ):
            path = self.assets / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_links_and_is_idempotent(self):
        setup(self.assets, self.repo)
        setup(self.assets, self.repo)
        self.assertTrue((self.repo / "VCTK_2F2M_train").is_symlink())
        self.assertTrue((self.repo / "log/log_Gunhee/vc_255.pt").is_symlink())
        self.assertEqual(
            (self.repo / "log/log_Gunhee/vc_255.pt").resolve(),
            (self.assets / "vc_255.pt").resolve(),
        )
        new_codebook = (
            self.repo
            / "log/codebook_stock_255_exclude/p236_exclude/codebook_stock_p236_512.pt"
        )
        new_codebook.touch()
        self.assertTrue(new_codebook.is_file())
        self.assertFalse(
            (
                self.assets
                / "log/codebook_stock_255_exclude/p236_exclude/codebook_stock_p236_512.pt"
            ).exists()
        )

    def test_does_not_overwrite_existing_asset(self):
        (self.repo / "checkpts").mkdir()
        with self.assertRaises(FileExistsError):
            setup(self.assets, self.repo)
        self.assertFalse((self.repo / "VCTK_2F2M_train").exists())

    def test_requires_complete_download(self):
        (self.assets / "vc_255.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            setup(self.assets, self.repo)
        self.assertFalse((self.repo / "VCTK_2F2M_train").exists())


if __name__ == "__main__":
    unittest.main()
