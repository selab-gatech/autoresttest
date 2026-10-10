import gzip
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gensim.models import KeyedVectors

from autoresttest.utils.utils import GLOVE_MODEL, SAVED_VECTORS_DIR, load_glove_vectors


class GloveVectorLoadingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base_dir = directory.name
        self.model_dir = Path(self.base_dir) / GLOVE_MODEL
        self.saved_dir = self.model_dir / SAVED_VECTORS_DIR
        for patcher in (
            patch("autoresttest.utils.utils.BASE_DIR", self.base_dir),
            patch(
                "autoresttest.utils.utils.load",
                side_effect=AssertionError("gensim downloader called"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_text_vectors(self):
        self.model_dir.mkdir()
        with gzip.open(self.model_dir / f"{GLOVE_MODEL}.gz", "wt") as fh:
            fh.write("2 3\nuser 0.1 0.2 0.3\nid 0.4 0.5 0.6\n")

    def test_cached_vectors_load_without_contacting_the_downloader(self):
        self.write_text_vectors()
        vectors = load_glove_vectors()
        self.assertEqual(vectors.vector_size, 3)
        self.assertAlmostEqual(float(vectors["id"][1]), 0.5, places=6)

    def test_later_loads_use_the_saved_vectors(self):
        self.write_text_vectors()
        load_glove_vectors()
        self.assertEqual(
            [p.name for p in self.model_dir.iterdir() if p.is_dir()],
            [SAVED_VECTORS_DIR],
        )
        with patch.object(
            KeyedVectors,
            "load_word2vec_format",
            side_effect=AssertionError("text vectors parsed again"),
        ):
            vectors = load_glove_vectors()
        self.assertIsInstance(vectors.vectors, np.memmap)
        self.assertEqual(stat.S_IMODE(self.saved_dir.stat().st_mode), 0o755)
        np.testing.assert_allclose(vectors["user"], [0.1, 0.2, 0.3], rtol=1e-6)

    def test_unreadable_saved_vectors_fall_back_to_the_text(self):
        self.write_text_vectors()
        load_glove_vectors()
        array = self.saved_dir / f"{GLOVE_MODEL}.kv.vectors.npy"
        array.write_bytes(array.read_bytes()[:-8])  # truncated
        with patch("builtins.print") as printed:
            vectors = load_glove_vectors()
        self.assertAlmostEqual(float(vectors["id"][1]), 0.5, places=6)
        self.assertIn("Could not load the saved vectors", printed.call_args.args[0])

    @unittest.skipIf(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        "file permissions do not stop root or apply on Windows",
    )
    def test_a_saved_folder_of_another_user_falls_back_to_the_text(self):
        self.write_text_vectors()
        load_glove_vectors()
        os.chmod(self.saved_dir, 0o000)
        self.addCleanup(os.chmod, self.saved_dir, 0o755)
        with patch("builtins.print") as printed:
            self.assertEqual(load_glove_vectors().vector_size, 3)
        self.assertIn("Permission denied", printed.call_args.args[0])

    @unittest.skipIf(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        "file permissions do not stop root or apply on Windows",
    )
    def test_a_read_only_cache_still_loads(self):
        self.write_text_vectors()
        os.chmod(self.model_dir, 0o555)
        self.addCleanup(os.chmod, self.model_dir, 0o755)
        self.assertEqual(load_glove_vectors().vector_size, 3)
        self.assertEqual(
            [p.name for p in self.model_dir.iterdir()], [f"{GLOVE_MODEL}.gz"]
        )

    def test_missing_cache_falls_back_to_the_downloader(self):
        downloaded = KeyedVectors(vector_size=2)
        downloaded.add_vectors(["id"], np.array([[0.5, 0.5]]))
        with patch("autoresttest.utils.utils.load", return_value=downloaded) as load:
            self.assertIs(load_glove_vectors(), downloaded)
        load.assert_called_once_with(GLOVE_MODEL)
        self.assertTrue((self.saved_dir / f"{GLOVE_MODEL}.kv").is_file())


if __name__ == "__main__":
    unittest.main()
