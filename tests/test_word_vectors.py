import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresttest.utils.utils import GLOVE_MODEL, load_glove_vectors


class GloveVectorLoadingTests(unittest.TestCase):
    def test_cached_vectors_load_without_contacting_the_downloader(self):
        with tempfile.TemporaryDirectory() as base_dir:
            model_dir = Path(base_dir) / GLOVE_MODEL
            model_dir.mkdir()
            with gzip.open(model_dir / f"{GLOVE_MODEL}.gz", "wt") as fh:
                fh.write("2 3\nuser 0.1 0.2 0.3\nid 0.4 0.5 0.6\n")
            with (
                patch("autoresttest.utils.utils.BASE_DIR", base_dir),
                patch(
                    "autoresttest.utils.utils.load",
                    side_effect=AssertionError("gensim downloader called"),
                ),
            ):
                vectors = load_glove_vectors()
        self.assertEqual(vectors.vector_size, 3)
        self.assertAlmostEqual(float(vectors["id"][1]), 0.5, places=6)

    def test_missing_cache_falls_back_to_the_downloader(self):
        with (
            tempfile.TemporaryDirectory() as base_dir,
            patch("autoresttest.utils.utils.BASE_DIR", base_dir),
            patch("autoresttest.utils.utils.load", return_value="vectors") as load,
        ):
            self.assertEqual(load_glove_vectors(), "vectors")
        load.assert_called_once_with(GLOVE_MODEL)


if __name__ == "__main__":
    unittest.main()
