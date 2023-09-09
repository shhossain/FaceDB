import unittest
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from facedb import (
    FaceDB,
)


class TestFaceDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = FaceDB(
            path="facedata",
            metric="euclidean",
            database_backend="chromadb",
            embedding_dim=128,
            module="face_recognition",
        )

    def test_add_many(self):
        files = [
            current_dir / "imgs" / "joe_biden.jpeg",
            current_dir / "imgs" / "no_face.png",
            current_dir / "imgs" / "narendra_modi.jpeg",
        ]
        imgs = []
        names = []
        for file in files:
            imgs.append(file)
            names.append(Path(file).name)

        ids, failed_indexes = self.db.add_many(imgs=imgs, names=names)

        self.assertEqual(len(failed_indexes), 1)
        self.assertEqual(len(ids), 2)

    def test_recognize_known_face(self):
        known_face = current_dir / "imgs" / "joe_biden_2.jpeg"
        result = self.db.recognize(img=known_face, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden", result["name"])  # type: ignore

    def test_recognize_unknown_face(self):
        unknown_face = current_dir / "imgs" / "barack_obama.jpeg"
        result = self.db.recognize(img=unknown_face, include=["name"])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
