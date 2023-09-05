from typing import Literal, Optional
import pinecone
import numpy as np
import chromadb
import cv2
import pprint
from typing import Union
from pathlib import Path
import os
import shutil


def many_vectors(obj):
    if isinstance(obj, list):
        if len(obj) == 0:
            return False
        elif isinstance(obj[0], list):
            return True
        elif isinstance(obj[0], np.ndarray):
            return True
    return False


class FaceResults(list["FaceResult"]):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        if len(self) == 1:
            self.name = self[0].name
            self.id = self[0].id
            self.distance = self[0].distance
            self.embedding = self[0].embedding
            self.img = self[0].img
            self.kw = self[0].kw

            for i in self.kw:
                setattr(self, i, self.kw[i])

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            if key == "name":
                return self.name
            elif key == "id":
                return self.id
            elif key == "distance":
                return self.distance
            elif key == "embedding":
                return self.embedding
            elif key == "img":
                return self.img
            else:
                return self.kw[key]

    def __repr__(self):
        return f"<FaceResults count={len(self)}>"

    def __str__(self):
        return pprint.pformat([i.__str__() for i in self])

    def show_img(self, per_row=5, limit=10, page=1, img_size=(100, 100)):
        import matplotlib.pyplot as plt

        if not self or self[0]["img"] is None:  # type: ignore
            print("No image available")
            return

        if len(self) > limit:
            self = self[limit * (page - 1) : limit * page]

        # resize
        for i in self:  # type: ignore
            i["img"] = cv2.resize(i["img"], img_size)
            i["img"] = cv2.cvtColor(i["img"], cv2.COLOR_BGR2RGB)

        num_images = len(self)  # type: ignore
        num_columns = 4
        images_per_row = per_row

        num_rows = int(np.ceil(num_images / images_per_row))

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_columns == 1:
            axes = np.expand_dims(axes, axis=1)

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(self[i]["img"])  # type: ignore
                ax.axis("off")  # type: ignore
            else:
                ax.axis("off")  # type: ignore

        plt.tight_layout()

        plt.show()


class FaceResult(dict):
    def __init__(self, id, name=None, distance=None, embedding=None, img=None, **kw):
        kw["id"] = id
        kw["name"] = name
        kw["distance"] = distance
        kw["embedding"] = embedding
        kw["img"] = img

        self.kw = kw

        for i in kw:
            setattr(self, i, kw[i])

        super().__init__(**kw)

    def __repr__(self):
        return f"<FaceResult id={self['id']} name={self['name']}>"

    def __str__(self):
        result = {}

        keys = list(self.keys())
        keys.remove("img")
        keys.remove("embedding")

        for key in keys:
            val = self[key]
            if val:
                result[key] = val

        return pprint.pformat(result)

    def show_img(self):
        if self.get("img") is None:
            print("No image available")
            return
        else:
            import matplotlib.pyplot as plt

            plt.imshow(self["img"])
            plt.show()


class BaseDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

    def add(self, ids, embeddings, metadatas=None):
        if isinstance(ids, str):
            ids = [ids]
        if not many_vectors(embeddings):
            embeddings = [embeddings]

        assert len(ids) == len(
            embeddings
        ), "ids and embeddings must have the same length"
        if metadatas:
            assert len(ids) == len(
                metadatas
            ), "ids and metadatas must have the same length"

            if not isinstance(metadatas, list):
                metadatas = [metadatas]

            assert isinstance(metadatas[0], dict), "metadatas must be a list of dict"

        return self._add(ids, embeddings, metadatas)

    def _add(self, ids, embeddings, metadatas=None):
        raise NotImplementedError

    def delete(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        return self._delete(ids)

    def _delete(self, ids):
        raise NotImplementedError

    def update(self, ids, embeddings=None, metadatas=None):
        if isinstance(ids, str):
            ids = [ids]
        if embeddings:
            if not many_vectors(embeddings):
                embeddings = [embeddings]

            assert len(ids) == len(
                embeddings
            ), "ids and embeddings must have the same length"
        if metadatas:
            if not isinstance(metadatas, list):
                metadatas = [metadatas]

            assert len(ids) == len(
                metadatas
            ), "ids and metadatas must have the same length"
            assert isinstance(metadatas[0], dict), "metadatas must be a list of dict"

        return self._update(ids, embeddings, metadatas)

    def _update(self, ids, embeddings=None, metadatas=None):
        raise NotImplementedError

    def query(self, embeddings, top_k=1, include=None, where=None):
        raise NotImplementedError

    def get(self, ids, include=None):
        raise NotImplementedError

    def parser(self, result, imgdb, include=None, query=True) -> list[FaceResults]:
        raise NotImplementedError

    def all(self, include=None):
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.count()


class PineconeDB(BaseDB):
    def __init__(
        self, dimension: int, index=None, index_name=None, metric="cosine", **kw
    ):
        self.index: pinecone.Index = None  # type: ignore
        self.dimension: int = dimension
        assert metric in [
            "cosine",
            "euclidean",
            "dotproduct",
        ], "metric must be cosine, euclidean, or dotproduct"
        assert (
            index is not None or index_name is not None
        ), "index or index_name must be provided"
        assert dimension is not None, "dimension must be provided"

        if not index:
            pinecone.init(
                api_key=kw.get("api_key", os.environ.get("PINECONE_API_KEY")),
                environment=kw.get(
                    "environment", os.environ.get("PINECONE_ENVIRONMENT")
                ),
            )

            self.index = self.get_index(index_name, dimension, metric)
        else:
            assert isinstance(
                index, pinecone.Index
            ), "index must be a pinecone.Index object"
            self.index = index

        self.index_name = self.index.configuration["index_name"]  # type: ignore

        self.index_info: dict = pinecone.describe_index(self.index_name)  # type: ignore

        assert (
            self.index_info["database"]["dimension"] == self.dimension  # type: ignore
        ), "dimension must be the same as the index"

        assert (
            self.index_info["database"]["metric"] == metric  # type: ignore
        ), "metric must be the same as the index"

    def count(self):
        return self.index_info["total_vector_count"]

    def get_index(self, index_name, dimension, metric):
        if index_name in pinecone.list_indexes():
            return pinecone.Index(index_name)
        else:
            pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
            return pinecone.Index(index_name)

    def _add(self, ids, embeddings, metadatas=None):
        vectors = []
        for i in range(len(ids)):
            data = {"id": ids[i], "values": embeddings[i]}
            if metadatas:
                data["metadata"] = metadatas[i]
            vectors.append(data)

        return self.index.upsert(vectors=vectors)

    def _delete(self, ids):
        return self.index.delete(ids)

    def delete_all(self):
        return self.index.delete(delete_all=True)

    def _update(self, ids, embeddings=None, metadatas=None):
        res = []
        for i in range(len(ids)):
            data = {
                "id": ids[i],
                "values": None,
                "set_metadata": None,
            }

            if embeddings:
                data["values"] = embeddings[i]
            if metadatas:
                data["set_metadata"] = metadatas[i]

            res.append(self.index.update(**data))

        return res

    def query(
        self,
        embeddings,
        top_k=1,
        include: Optional[list[Literal["embeddings", "metadatas"]]] = None,
        where=None,
    ):
        params = {
            "top_k": top_k,
        }

        if include is not None:
            if isinstance(include, str):
                include = [include]  # type: ignore

            for i in include:  # type: ignore
                if i == "embeddings":
                    params["include_values"] = True
                elif i == "metadatas":
                    params["include_metadata"] = True

        if where:
            params["filter"] = where

        if many_vectors(embeddings):
            res = []
            for i in range(len(embeddings)):
                res.append(self.index.query(vector=embeddings[i], **params).to_dict())  # type: ignore
            return res
        else:
            return self.index.query(vector=embeddings, **params).to_dict()  # type: ignore

    def parser(self, result, imgdb, include=None, query=True) -> list[FaceResults]:
        if isinstance(result, dict):
            result = [result]

        if query:
            key = "matches"
        else:
            key = "vectors"

        if include is None:
            include = []
        elif isinstance(include, str):
            include = [include]

        results: list[FaceResults] = []

        for i in range(len(result)):
            rs = []
            for j, r in enumerate(result[i][key]):
                data = {
                    "id": r["id"],
                }

                if "score" in r:
                    data["distance"] = 1 - r["score"]

                for k in include:
                    if k[:9] == "embedding":
                        data["embedding"] = r["values"]
                    elif k[:3] == "img":
                        data["img"] = imgdb.get(r["id"])
                    elif k[:8] == "distance":
                        continue
                    else:
                        if "metadata" in r and r["metadata"]:
                            try:
                                data[k] = r["metadata"][k]
                            except KeyError:
                                data[k] = None

                rs.append(FaceResult(**data))
            if rs:
                results.append(FaceResults(rs))
            else:
                results.append(None)  # type: ignore

        return results

    def get(self, ids, include=None, where=None):
        return self.index.fetch(ids=ids).to_dict()

    def all(self, include=None):
        get_metadata = None
        get_values = None
        if include:
            if "metadatas" in include:
                get_metadata = True
            if "embeddings" in include:
                get_values = True

        stats = self.index.describe_index_stats()
        namespace_map = stats["namespaces"]
        ret = []
        for namespace in namespace_map:
            vector_count = namespace_map[namespace]["vector_count"]
            res = self.index.query(
                vector=[0 for _ in range(self.dimension)],
                top_k=vector_count,
                namespace=namespace,
                include_values=get_values,
                include_metadata=get_metadata,
            )
            ret.extend(res["matches"])

        return {"vectors": ret}


class ChromaDB(BaseDB):
    def __init__(
        self, path=None, client=None, metric="cosine", collection_name="faces"
    ):
        assert metric in [
            "cosine",
            "l2",
            "ip",
        ], "chromadb only support cosine, l2, and ip metric"

        if path is None:
            path = "data"
        if client is None:
            client = chromadb.PersistentClient(path)

        self.db = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": metric,
            },
        )

    def add(self, ids, embeddings, metadatas=None):
        return self.db.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def delete(self, ids):
        return self.db.delete(ids)

    def update(self, ids, embeddings=None, metadatas=None):
        return self.db.update(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        embeddings,
        top_k=1,
        include: Optional[list[Literal["embeddings", "metadatas", "distances"]]] = [
            "distances"
        ],
        where=None,
    ):
        return self.db.query(
            query_embeddings=embeddings, n_results=top_k, include=include or ["distances"], where=where  # type: ignore
        )

    def get(self, ids, include=None, where=None):
        return self.db.get(ids=ids, include=include or ["metadatas"], where=where)

    def __len__(self):
        return self.db.count()

    def all(self, include=None):
        return self.db.get(include=include or ["metadatas"])

    def query_parser(self, result, imgdb, include=["distances"]) -> list[FaceResults]:
        results: list[FaceResults] = []
        for i in range(len(result["ids"])):
            rs = []
            for j, id in enumerate(result["ids"][i]):
                data = {"id": id}
                if result["distances"]:
                    data["distance"] = result["distances"][i][j]
                for k in include:
                    if k[:9] == "embedding":
                        if result["embeddings"]:
                            data["embedding"] = result["embeddings"][i][j]
                    elif k[:3] == "img":
                        data["img"] = imgdb.get(id)
                    else:
                        if (
                            result["metadatas"]
                            and result["metadatas"][i]
                            and result["metadatas"][i][j]
                        ):
                            try:
                                data[k] = result["metadatas"][i][j][k]
                            except KeyError:
                                data[k] = None
                rs.append(FaceResult(**data))
            if rs:
                results.append(FaceResults(rs))
            else:
                results.append(None)  # type: ignore
        return results

    def get_parser(self, result, imgdb, include=["metadatas"]) -> FaceResults:
        if include is None:
            include = []
        elif isinstance(include, str):
            include = [include]

        results: list[FaceResult] = []
        for i, id in enumerate(result["ids"]):
            data = {"id": id}
            for k in include:
                if k[:9] == "embedding":
                    data["embedding"] = result["embeddings"][i]
                elif k[:3] == "img":
                    data["img"] = imgdb.get(id)
                else:
                    if result["metadatas"] and result["metadatas"][i]:
                        try:
                            data[k] = result["metadatas"][i][k]
                        except KeyError:
                            data[k] = None
            results.append(FaceResult(**data))

        return FaceResults(results)

    def parser(self, result, imgdb, include=None, query=True):
        if query:
            return self.query_parser(result, imgdb, include)
        else:
            return self.get_parser(result, imgdb, include)

    def delete_all(self):
        shutil.rmtree(str(self.path))
