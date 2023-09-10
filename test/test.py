import os
import unittest
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from facedb import (
    FaceDB,
)


class TestFaceDBChroma(unittest.TestCase):
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

        self.assertEqual(len(failed_indexes), 1)
        self.assertEqual(len(ids), 2)

    def test_recognize_known_face(self):
        known_face = str(current_dir / "imgs" / "joe_biden_2.jpeg")
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
        idx = self.db.recognize(img=img, include=["name"]).id  # type: ignore
        self.db.update(id=idx, name="joe_biden_2")

        result = self.db.recognize(img=img, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden_2", result["name"])  # type: ignore

    def test_get(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
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
        if result is None:
            self.assertIsNone(result)
        else:
            self.assertEqual(len(result), 0)

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
        

class TestFaceDBPinecone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = FaceDB(
            path="facedata",
            metric="euclidean",
            database_backend="pinecone",
            embedding_dim=128,
            module="face_recognition",
            index_name="test-face-db"
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

        print(
            f"Failed indexes: {failed_indexes}\n"
            f"IDs: {ids}"
        )
        print(self.db.all(include=["name"]))
        
        self.assertEqual(len(failed_indexes), 1)
        self.assertEqual(len(ids), 2)

    def test_recognize_known_face(self):
        known_face = str(current_dir / "imgs" / "joe_biden_2.jpeg")
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
        idx = self.db.recognize(img=img, include=["name"]).id  # type: ignore
        self.db.update(id=idx, name="joe_biden_2")

        result = self.db.recognize(img=img, include=["name"])
        self.assertIsNotNone(result)
        if result:
            self.assertIn("joe_biden_2", result["name"])  # type: ignore

    def test_get(self):
        img = current_dir / "imgs" / "joe_biden_2.jpeg"
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
        if result is None:
            self.assertIsNone(result)
        else:
            self.assertEqual(len(result), 0)

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
    suite = unittest.TestSuite()
    suite.addTest(TestFaceDBChroma("test_add_many"))
    suite.addTest(TestFaceDBChroma("test_recognize_known_face"))
    suite.addTest(TestFaceDBChroma("test_recognize_unknown_face"))
    suite.addTest(TestFaceDBChroma("test_update"))
    suite.addTest(TestFaceDBChroma("test_get"))
    suite.addTest(TestFaceDBChroma("test_search"))
    suite.addTest(TestFaceDBChroma("test_delete"))
    suite.addTest(TestFaceDBChroma("test_query"))
    
    # get api_key and env from sys
    api_key = sys.argv[1]
    env = sys.argv[2]
    
    os.environ["PINECONE_API_KEY"] = api_key
    os.environ["PINECONE_ENVIRONMENT"] = env
    
    suite.addTest(TestFaceDBPinecone("test_add_many"))
    suite.addTest(TestFaceDBPinecone("test_recognize_known_face"))
    suite.addTest(TestFaceDBPinecone("test_recognize_unknown_face"))
    suite.addTest(TestFaceDBPinecone("test_update"))
    suite.addTest(TestFaceDBPinecone("test_get"))
    suite.addTest(TestFaceDBPinecone("test_search"))
    suite.addTest(TestFaceDBPinecone("test_delete"))
    suite.addTest(TestFaceDBPinecone("test_query"))

    # Run the test suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
