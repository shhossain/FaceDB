import numpy as np
import chromadb
import cv2
import pprint
from pathlib import Path
import os
from facedb.query import Query
from math import ceil

try:
    from typing import Literal, Optional, Union, List
except ImportError:
    from typing_extensions import Literal, Optional, Union, List

pinecone = None


def many_vectors(obj):
    if isinstance(obj, list):
        if len(obj) == 0:
            return False
        elif isinstance(obj[0], list):
            return True
        elif isinstance(obj[0], np.ndarray):
            return True
    return False


def calculate_confidence(dis, threshold, direction, assume=80):
    if direction == "positive":
        return ceil((assume / (threshold)) * (dis))
    elif direction == "negative":
        return ceil((assume / (1 - threshold)) * (1 - dis))


class FaceResults(List["FaceResult"]):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        if len(self) == 1:
            self.name = self[0].name  # type: ignore
            self.id = self[0].id
            self.distance = self[0].distance
            self.embedding = self[0].embedding
            self.img = self[0].img
            self.kw = self[0].kw

            for i in self.kw:
                setattr(self, i, self.kw[i])

    def query(self, *args, **kw):
        q = {}
        for i in args:
            q.update(i)
        q.update(kw)

        query = Query(q)
        results = list(filter(query.match, self))  # type: ignore
        return FaceResults(results)

    def query_generator(self, query):
        query = Query(query)
        result = filter(query.match, self)  # type: ignore
        for i in result:
            yield i

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
        txt = f"FaceResults [\n"
        ct = 0
        for i in self:
            t = f"id={i['id']} name={i['name']}"
            if i.get("confidence"):
                t += f" confidence={i['confidence']}%"
            t += ",\n"
            txt += t
            if ct == 5:
                txt += f"... {len(self) - ct} more"
                break
            ct += 1
        if txt.endswith(",\n"):
            txt = txt[:-2]

        txt += " ]"
        return txt

    def __str__(self):
        return self.__repr__()

    @property
    def df(self):
        import pandas as pd

        data = []
        for i in self:
            d = {}
            for k in i:
                val = i[k]
                if val is not None:
                    if k == "embedding":
                        val = f"Embedding({len(val)} dim)"
                    elif k == "img":
                        val = "Image(cv2 image)"
                    d[k] = val
            data.append(d)

        return pd.DataFrame(data)

    def show_img(self, limit=10, page=1, img_size=(100, 100)):
        if len(self) == 0:
            print("No image available")
            return

        if page < 1:
            page = 1
        elif page > len(self) // limit + 1:
            page = len(self) // limit + 1

        images = []
        for i in range((page - 1) * limit, min(page * limit, len(self))):
            images.append({"name": self[i].name, "image": self[i].img})  # type: ignore

        fixed_size = (200, 200)
        for item in images:
            if item["image"] is None:
                # create a blank image white color
                img = np.zeros((fixed_size[0], fixed_size[1], 3), dtype=np.uint8)
                # put not img available text in middle
                cv2.putText(
                    img,
                    "No image available",
                    (10, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                item["image"] = img
            else:
                # resize image
                item["image"] = cv2.resize(item["image"], fixed_size)

        # Define text settings
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.3
        font_thickness = 1

        # Create a blank canvas
        num_rows = 2
        num_cols = 2
        canvas_width = fixed_size[0] * num_cols
        canvas_height = fixed_size[1] * num_rows
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Populate the canvas with images and their names
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                if index < len(images):
                    y_offset = i * fixed_size[1]
                    x_offset = j * fixed_size[0]

                    # Add the image to the canvas
                    canvas[
                        y_offset : y_offset + fixed_size[1],
                        x_offset : x_offset + fixed_size[0],
                    ] = images[index]["image"]

                    # Add a blurred background for the name
                    name = images[index]["name"]

                    # add name in bottom left corner
                    text_size, _ = cv2.getTextSize(
                        name, font, font_scale, font_thickness
                    )
                    text_x = x_offset + 5
                    text_y = y_offset + fixed_size[1] - 20

                    avg_color = np.average(images[index]["image"], axis=(0, 1))
                    text_color = (
                        (255, 255, 255) if np.mean(avg_color) < 128 else (0, 0, 0)
                    )

                    # Create a region of interest (ROI) for the text background
                    roi = canvas[
                        text_y : text_y + text_size[1] + 5,
                        text_x : text_x + text_size[0] + 5,
                    ]

                    # Apply Gaussian blur to the ROI
                    roi = cv2.GaussianBlur(roi, (15, 15), 0)

                    # Place the blurred ROI back onto the canvas
                    canvas[
                        text_y : text_y + text_size[1] + 5,
                        text_x : text_x + text_size[0] + 5,
                    ] = roi

                    # Add the image name as text annotation in the bottom right corner
                    cv2.putText(
                        canvas,
                        name,
                        (text_x, text_y + text_size[1]),
                        font,
                        font_scale,
                        text_color,
                        font_thickness,
                    )

        # Display the canvas
        import matplotlib.pyplot as plt

        plt.imshow(canvas)
        plt.show()


class FaceResult(dict):
    def __init__(self, id, name=None, distance=None, embedding=None, img=None, **kw):
        kw["id"] = id
        kw["name"] = name
        kw["distance"] = distance
        kw["embedding"] = embedding
        kw["img"] = img

        self.id = id
        self.name = name
        self.distance = distance
        self.embedding = embedding
        self.img = img

        self.kw = kw

        for i in kw:
            setattr(self, i, kw[i])

        super().__init__(**kw)

    def __repr__(self):
        txt = f"FaceResult(id={self.id}, name={self.name}"
        if self.get("confidence"):
            txt += f", confidence={self['confidence']}%"
        txt += ")"
        return txt

    def __str__(self):
        result = {}
        for key in self:
            val = self[key]
            if val is not None:
                if key == "embedding":
                    val = f"Embedding({len(val)} dim)"
                elif key == "img":
                    val = "Image(cv2 image)"
                result[key] = val

        return pprint.pformat(result)

    def show_img(self):
        if self.get("img") is None:
            print("No image available")
            return
        else:
            import matplotlib.pyplot as plt

            if self.img is None:
                print("No image available. Include `img` in `include` to get the image")
                return

            plt.imshow(self.img)  # type: ignore
            plt.title(self.name)  # type: ignore
            plt.show()


class BaseDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __str__(self):
        return self.__repr__()

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

    def _get(self, ids, include=None, where=None):
        raise NotImplementedError

    def get(self, ids, include=None, where=None):
        if isinstance(ids, str):
            ids = [ids]
        return self._get(ids, include, where)

    def parser(self, result, imgdb, include=None, query=True) -> List[FaceResults]:
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
        global pinecone
        if pinecone is None:
            try:
                import pinecone
            except ImportError:
                raise ImportError(
                    "pinecone is not installed. Install it with `pip install pinecone-client`"
                )

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
            api_key = kw.get("api_key", None) or os.environ.get(
                "PINECONE_API_KEY", None
            )
            environment = kw.get(
                "environment",
            ) or os.environ.get("PINECONE_ENVIRONMENT", None)

            if api_key is None or environment is None:
                raise ConnectionError(
                    "Pinecone api_key and environment must be provided. Please see https://www.pinecone.io/docs/quick-start/ for more information"
                )
            try:
                pinecone.init(  # type: ignore
                    api_key=api_key,
                    environment=environment,
                )
                api_key = None
                environment = None
                del api_key
                del environment
            except Exception as e:
                raise Exception(
                    f"Failed to initialize pinecone. Please check your api_key and environment. Error: {e}"
                )
            self.index = self.get_index(index_name, dimension, metric)
        else:
            assert isinstance(
                index, pinecone.Index  # type: ignore
            ), f"index must be a pinecone.Index object, got `{type(index)}`"
            self.index = index

        self.index_name = self.index.configuration.server_variables["index_name"]

        self.index_info: dict = pinecone.describe_index(self.index_name)  # type: ignore

        assert (
            self.index_info.dimension == self.dimension  # type: ignore
        ), f"dimension must be the same as the index. `{self.index_name}` has dimension of `{self.index_info.dimension}` but got `{self.dimension}`"

        assert (
            self.index_info.metric == metric  # type: ignore
        ), f"metric must be the same as the index. `{self.index_name}` has metric of `{self.index_info.metric}` but got `{metric}`"

    def count(self):
        return self.index.describe_index_stats().get("total_vector_count", -1)

    def get_index(self, index_name, dimension, metric):
        if index_name in pinecone.list_indexes():  # type: ignore
            return pinecone.Index(index_name)  # type: ignore
        else:
            pinecone.create_index(name=index_name, dimension=dimension, metric=metric)  # type: ignore
            return pinecone.Index(index_name)  # type: ignore

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
        include: Optional[List[Literal["embeddings", "metadatas"]]] = None,
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

    def parser(
        self, result, imgdb, include=None, query=True, threshold=None
    ) -> Union[FaceResults, List[FaceResults]]:
        if isinstance(result, dict):
            result = [result]

        if include is None:
            include = []
        elif isinstance(include, str):
            include = [include]

        results: List[FaceResults] = []

        for i in range(len(result)):
            rs = []
            for j, r in enumerate(result[i]["matches"]):
                data = {
                    "id": r["id"],
                }
                if "score" in r:
                    data["distance"] = 1 - r["score"]
                    if threshold:
                        _, direction, value = threshold
                        data["confidence"] = calculate_confidence(
                            data["distance"], value, direction
                        )

                for k in include:
                    if k[:9] == "embedding":
                        data["embedding"] = r["values"]
                    elif k[:3] == "img":
                        data["img"] = imgdb.get(r["id"])
                    elif k[:8] == "distance":
                        continue

                # add all metadata keys
                if "metadata" in r and r["metadata"]:
                    for key in r["metadata"]:
                        if key == "values" or key == "id" or key == "score":
                            continue
                        data[key] = r["metadata"][key]

                rs.append(FaceResult(**data))
            if rs:
                results.append(FaceResults(rs))
            else:
                results.append(None)  # type: ignore

        if not query:
            return results[0]

        return results

    def get(self, ids, include=None, where=None):
        if isinstance(ids, str):
            ids = [ids]

        return self.query(
            embeddings=[0] * self.dimension,
            top_k=len(ids),
            where={"id": {"$in": ids}},
            include=include,
        )

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

        return {"matches": ret}


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
            self.client = chromadb.PersistentClient(path)
        else:
            self.client = client

        self.path = Path(path)

        self.db = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": metric,
            },
        )

        self.metric = metric
        self.collection_name = collection_name

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
        include: Optional[List[Literal["embeddings", "metadatas", "distances"]]] = [
            "distances"
        ],
        where=None,
    ):
        return self.db.query(
            query_embeddings=embeddings, n_results=top_k, include=include or ["distances"], where=where  # type: ignore
        )

    def get(self, ids, include=None, where=None):
        return self.db.get(ids=ids, include=include or ["metadatas"], where=where)

    def count(self) -> int:
        return self.db.count()

    def all(self, include=None):
        return self.db.get(include=include or ["metadatas"])

    def query_parser(
        self, result, imgdb, include=["distances"], threshold=None
    ) -> List[FaceResults]:
        results: List[FaceResults] = []
        for i in range(len(result["ids"])):
            rs = []
            for j, id in enumerate(result["ids"][i]):
                data = {"id": id}
                if result["distances"]:
                    data["distance"] = result["distances"][i][j]
                    if threshold:
                        _, direction, value = threshold
                        data["confidence"] = calculate_confidence(
                            data["distance"], value, direction
                        )

                for k in include:
                    if k[:9] == "embedding":
                        if result["embeddings"]:
                            data["embedding"] = result["embeddings"][i][j]
                    elif k[:3] == "img":
                        data["img"] = imgdb.get(id)

                # add all metadata keys
                if (
                    result["metadatas"]
                    and result["metadatas"][i]
                    and result["metadatas"][i][j]
                ):
                    for key in result["metadatas"][i][j]:
                        data[key] = result["metadatas"][i][j][key]
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

        results: List[FaceResult] = []
        for i, id in enumerate(result["ids"]):
            data = {"id": id}
            for k in include:
                if k[:9] == "embedding":
                    data["embedding"] = result["embeddings"][i]
                elif k[:3] == "img":
                    data["img"] = imgdb.get(id)

            # add all metadata keys
            if result["metadatas"] and result["metadatas"][i]:
                for key in result["metadatas"][i]:
                    data[key] = result["metadatas"][i][key]

            results.append(FaceResult(**data))

        return FaceResults(results)

    def parser(
        self, result, imgdb, include=None, query=True, threshold=None
    ) -> Union[FaceResults, List[FaceResults]]:
        if query:
            return self.query_parser(result, imgdb, include, threshold)
        else:
            return self.get_parser(result, imgdb, include)

    def delete_all(self):
        self.client.delete_collection(self.collection_name)
