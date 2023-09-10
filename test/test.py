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
            names.append(Path(file).stem)

        ids, failed_indexes = self.db.add_many(imgs=imgs, names=names)
        print(self.db.all(include=["name"]))

        self.assertEqual(len(failed_indexes), 1)
        self.assertEqual(len(ids), 2)

    def test_recognize_known_face(self):
        known_face = str(current_dir / "imgs" / "joe_biden_2.jpeg")
        print("1 Path:", known_face)
        result = self.db.recognize(img=known_face, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden", result["name"])  # type: ignore

    def test_recognize_unknown_face(self):
        unknown_face = current_dir / "imgs" / "barak_obama.jpeg"
        result = self.db.recognize(img=unknown_face, include=["name"])
        self.assertIsNone(result)

    def test_update(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
        print("Path:", img)
        idx = self.db.recognize(img=img, include=["name"]).id  # type: ignore
        self.db.update(id=idx, name="joe_biden_2")

        result = self.db.recognize(img=img, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden_2", result["name"])  # type: ignore

    def test_get(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
        print("Path:", img)
        idx = self.db.recognize(img=img, include=["name"]).id  # type: ignore
        result = self.db.get(id=idx, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden_2", result["name"])  # type: ignore

    def test_delete(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
        idx = self.db.recognize(img=img, include=["name"]).id  # type: ignore
        self.db.delete(id=idx)
        result = self.db.get(id=idx, include=["name"])
        self.assertIsNone(result)

    def test_search(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
        emb = self.db.embedding_func(img)
        result = self.db.search(embedding=emb, include=["name"])
        self.assertIsNotNone(result)

    def test_query(self):
        results = self.db.query(name="narendra_modi", include=["name"])
        self.assertIsNotNone(results)
        if results:
            self.assertEqual(results[0]["name"], "narendra_modi")  # type: ignore

    @classmethod
    def tearDownClass(cls):
        cls.db.delete_all()

if __name__ == "__main__":
    unittest.main()
