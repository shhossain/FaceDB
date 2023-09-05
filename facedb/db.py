import numpy as np
from tqdm.auto import tqdm
from typing import Literal, Callable, Optional, Union
import cv2
import warnings
import threading


DeepFace = None
deepface_distance = None
face_recognition = None

import_lock = threading.Lock()


def load_module(module: Literal["deepface", "face_recognition"]):
    with import_lock:
        if module == "deepface":
            global DeepFace
            global deepface_distance
            if DeepFace is None:
                try:
                    from deepface import DeepFace
                    from deepface.commons import distance as deepface_distance
                except ImportError:
                    raise ImportError(
                        "Please install `deepface` to use `deepface` module."
                    )
        elif module == "face_recognition":
            global face_recognition
            if face_recognition is None:
                try:
                    import face_recognition
                except ImportError:
                    raise ImportError(
                        "Please install `face_recognition` to use `face_recognition` module."
                    )
        else:
            raise ValueError(
                "Currently only `deepface` and `face_recognition` are supported."
            )


from facedb.db_tools import (
    get_embeddings,
    get_include,
    ImgDB,
    Rect,
    img_to_cv2,
    face_recognition_is_match,
    many_imgs,
    many_vectors,
    time_now,
    deeface_metric_map,
    get_model_dimension,
    l2_normalize,
)

from facedb.db_models import BaseDB, FaceResult, FaceResults, PineconeDB, ChromaDB

from pathlib import Path


def create_deepface_embedding_func(
    model_name,
    detector_backend,
    enforce_detection,
    align,
    normalization,
    l2_normalization,
):
    def embedding_func(img, enforce_detection=enforce_detection, **kw):
        try:
            result = DeepFace.represent(  # type: ignore
                img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                normalization=normalization,
            )
        except ValueError:
            return []

        result = [i["embedding"] for i in result]
        if l2_normalization:
            result = l2_normalize(result)
        return result

    return embedding_func


def create_face_recognition_embedding_func(
    model,
    num_jitters,
    l2_normalization,
):
    def embedding_func(img, know_face_locations=None, **kw):
        img = img_to_cv2(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = face_recognition.face_encodings(  # type: ignore
            img,
            num_jitters=num_jitters,
            model=model,
            known_face_locations=know_face_locations,
        )
        if result is None:
            return []

        if l2_normalization:
            result = l2_normalize(result)
        return result

    return embedding_func


def create_deepface_extract_faces_func(
    extract_faces_detector_backend,
    enforce_detection,
    align,
):
    def extract_faces(img):
        try:
            result = DeepFace.extract_faces(  # type: ignore
                img,
                detector_backend=extract_faces_detector_backend,
                enforce_detection=enforce_detection,
                align=align,
            )
        except ValueError:
            return []

        if result is None:
            return []

        return [Rect(**i["facial_area"]) for i in result]

    return extract_faces


def create_face_recognition_extract_faces_func(extract_face_model="hog"):
    def extract_faces(img):
        img = img_to_cv2(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = face_recognition.face_locations(img, model=extract_face_model)  # type: ignore
        if result is None:
            return []

        rects = []
        for i in result:
            y, xw, yh, x = i
            w = xw - x  # type: ignore
            h = yh - y  # type: ignore
            rects.append(Rect(x, y, w, h))
        return rects

    return extract_faces


metric_map = {
    "pinecone": {
        "cosine": "cosine",
        "euclidean": "euclidean",
        "dot": "dotproduct",
    },
    "chromadb": {
        "cosine": "cosine",
        "euclidean": "l2",
        "dot": "ip",
    },
}


class FaceDB:
    def __init__(
        self,
        *,
        path=None,
        metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        embedding_func=None,
        embedding_dim: Optional[int] = None,
        l2_normalization: bool = True,
        module: Literal["deepface", "face_recognition"] = "deepface",
        database_backend: Literal["chromadb", "pinecone"] = "chromadb",
        **kw,
    ):
        if path is None:
            path = "data"

        path = Path(path)

        assert metric in ["cosine", "euclidean", "dot"], "Supported metrics are `cosine`, `euclidean` and `dot`."
        assert module in ["deepface", "face_recognition"], "Supported modules are `deepface` and `face_recognition`."
        assert database_backend in ["chromadb", "pinecone"], "Supported database backends are `chromadb` and `pinecone`."
        
        self.metric = metric_map[database_backend][metric]
        self.embedding_func: Callable = None  # type: ignore
        self.extract_faces: Callable = None  # type: ignore
        self.module = module
        self.db_backend = database_backend
        self.l2_normalization = l2_normalization
        load_module(module)

        self.deepface_model_name = kw.get("model_name", "Facenet")

        if embedding_func is None:
            if module == "deepface":
                self.embedding_func = create_deepface_embedding_func(
                    model_name=kw.pop("model_name", "Facenet"),
                    detector_backend=kw.pop("detector_backend", "ssd"),
                    enforce_detection=kw.pop("enforce_detection", True),
                    align=kw.pop("align", True),
                    normalization=kw.pop("normalization", "base"),
                    l2_normalization=l2_normalization,
                )
                self.extract_faces = create_deepface_extract_faces_func(
                    extract_faces_detector_backend=kw.pop(
                        "extract_face_backend",
                        "ssd",
                    ),
                    enforce_detection=kw.pop("enforce_detection", True),
                    align=kw.pop("align", True),
                )
            elif module == "face_recognition":
                self.embedding_func = create_face_recognition_embedding_func(
                    model=kw.pop("model", "small"),
                    num_jitters=kw.pop("num_jitters", 1),
                    l2_normalization=l2_normalization,
                )
                self.extract_faces = create_face_recognition_extract_faces_func(
                    extract_face_model=kw.pop("extract_face_model", "hog"),
                )

            else:
                raise ValueError(
                    "Currently only `deepface` and `face_recognition` are supported."
                )

        if metric != "cosine":
            warnings.warn(
                "Only `cosine` space is tested. Other spaces may not work as expected."
            )

        if database_backend == "chromadb":
            self.db = ChromaDB(
                path=str(path),
                client=kw.pop("client", None),
                metric=self.metric,
                collection_name=kw.pop("collection_name", "facedb"),
            )

        elif database_backend == "pinecone":
            self.db = PineconeDB(
                index=kw.pop("index", None),
                index_name=kw.pop("index_name", "facedb"),
                metric=self.metric,
                dimension=embedding_dim
                or get_model_dimension(module, self.deepface_model_name),
            )

        if not path.exists():
            path.mkdir(parents=True)

        self.imgdb = ImgDB(db_path=str(path / "img.db"))

    def __len__(self):
        return self.db.count()

    def count(self):
        return self.db.count()

    def _is_match(self, distance, threshold=None):
        if self.module == "deepface":
            threshold = deepface_distance.findThreshold(  # type: ignore
                self.deepface_model_name, self.metric
            )
            if distance <= threshold:
                return True
        elif self.module == "face_recognition":
            if face_recognition_is_match(
                db_backend=self.db_backend,
                metric=self.metric,
                value=distance,
                threshold=threshold,
                l2_normalization=self.l2_normalization,
            ):
                return True

        return False

    def get_faces(self, img, *, zoom_out=0.25, only_rect=False) -> Union[None, list]:
        img = img_to_cv2(img)
        rects = self.extract_faces(img)
        img_h, img_w = img.shape[:2]
        for rect in rects:
            rect.x = max(0, rect.x - int(rect.w * zoom_out))
            rect.y = max(0, rect.y - int(rect.h * zoom_out))
            rect.w = min(img_w - rect.x, rect.w + int(rect.w * zoom_out))
            rect.h = min(img_h - rect.y, rect.h + int(rect.h * zoom_out))
        if rects:
            if only_rect:
                return rects

            return [
                img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
                for rect in rects
            ]
        return None

    def check_similar(self, embeddings, threshold=None) -> list:
        embeddings = get_embeddings(
            imgs=None,
            embeddings=embeddings,
        )

        result = self.db.query(
            embeddings=embeddings,
            top_k=1,
            include=None,
        )
        results = self.db.parser(result, imgdb=self.imgdb, include=["distances"])
        rs = []
        for result in results:
            if result is None:
                rs.append(False)
            elif self._is_match(result["distance"], threshold):
                rs.append(result["id"])
            else:
                rs.append(False)
        return rs

    def recognize(
        self, *, img=None, embedding=None, include=None, threshold=None, top_k=1
    ):
        single = False
        if embedding:
            if not many_vectors(embedding):
                single = True
        elif img:
            if not many_imgs(img):
                single = True

        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        rincludes, include = get_include(default="distances", include=include)
        result = self.db.query(
            embeddings=embedding,
            top_k=top_k,
            include=rincludes,
        )

        results = self.db.parser(result, imgdb=self.imgdb, include=include)
        res = []
        for result in results:
            if result is None:
                res.append(None)
            elif self._is_match(result["distance"], threshold):
                res.append(result)
            else:
                res.append(None)

        if single and res:
            return res[0]

        return res

    def add(
        self,
        name,
        img=None,
        embedding=None,
        id=None,
        check_similar=True,
        save_just_face=False,
        **metadata,
    ) -> str:
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        if check_similar:
            result = self.check_similar(embeddings=embedding)[0]
            if result:
                print(
                    "Similar face already exists. If you want to add anyway, set `check_similar` to `False`."
                )
                return result

        metadata["name"] = name
        idx = id or name + "-" + time_now()
        if img is not None:
            if save_just_face:
                img = self.get_faces(img)
                if img is None:
                    raise ValueError("No face found in the img.")
                img = img[0]

            self.imgdb.add(img_id=idx, img=img)

        self.db.add(
            ids=[idx],
            embeddings=embedding,
            metadatas=[
                {
                    **metadata,
                }
            ],
        )

        return idx

    def add_many(
        self,
        *,
        embeddings=None,
        imgs=None,
        metadata=None,
        ids=None,
        names=None,
        check_similar=True,
    ) -> tuple[list, list]:
        faces = []
        failed = []
        metadata_posible = metadata is not None
        if embeddings is None:
            if metadata_posible:
                warnings.warn(
                    "Without embeddings, add metadata may not work as expected."
                )
            if imgs is None:
                raise ValueError("Either `embeddings` or `imgs` must be provided.")

            if names is not None:
                if len(imgs) != len(names):
                    raise ValueError("`imgs` length and `names` length must be same")
                idxs = ids or [f"{name}-{time_now()}" for name in names]
            else:
                names = [f"face_{i}-{time_now()}" for i in range(len(imgs))]
                idxs = ids or [f"faceid_{i}-{time_now()}" for i in range(len(imgs))]

            for i, img in enumerate(tqdm(imgs, desc="Extracting faces")):
                try:
                    rects = self.get_faces(img, only_rect=True)
                    if not rects:
                        print(f"No face found in the img {i}. Skipping.")
                        failed.append(i)
                        continue
                    result = self.embedding_func(
                        img,
                        know_face_locations=[r.to(self.module) for r in rects],
                        enforce_detection=False,
                    )
                    try:
                        idx = idxs[i]
                    except IndexError:
                        raise ValueError("`ids` length and `imgs` length must be same")

                    if len(result) > 1:
                        # metadata_posible = False
                        for j, embedding in enumerate(result):
                            name = f"{names[i]}_{j}"
                            faces.append(
                                {
                                    "id": f"{idx}_{j}",
                                    "name": name,
                                    "embedding": embedding,
                                    "img": img,
                                    "rect": rects[j],
                                    "index": i,
                                }
                            )
                    elif len(result) > 0:
                        res = {
                            "id": idx,
                            "name": names[i],
                            "embedding": result[0],
                            "img": img,
                            "rect": rects[0],
                            "index": i,
                        }
                        if metadata_posible:
                            try:
                                res.update(metadata[i])
                            except IndexError:
                                pass
                        faces.append(res)
                    else:
                        warnings.warn(f"No face found in the img {i}. Skipping.")
                        failed.append(i)
                        continue
                except Exception as e:
                    print(f"Error in img {i}. Skipping.", e)
                    failed.append(i)
                    continue

            rects = None
            result = None
        else:
            idxs = ids or [f"faceid_{i}-{time_now()}" for i in range(len(embeddings))]
            for i, embedding in enumerate(embeddings):
                try:
                    idx = idxs[i]
                except IndexError:
                    raise ValueError(
                        "`ids` length and `embeddings` length must be same"
                    )

                res = {
                    "id": idx,
                    "embedding": embedding,
                    "index": i,
                }

                if names is not None:
                    try:
                        res["name"] = names[i]
                    except IndexError:
                        raise ValueError(
                            "`names` length and `embeddings` length must be same"
                        )
                else:
                    res["name"] = f"face_{i}-{time_now()}"

                if imgs is not None:
                    try:
                        rects = self.extract_faces(imgs[i])
                    except IndexError:
                        raise ValueError(
                            "`imgs` length and `embeddings` length must be same"
                        )

                    if len(rects) > 1:
                        print("Multiple faces found in the img. Taking first face.")

                    res["img"] = imgs[i]
                    res["rect"] = rects[0]

                if metadata_posible:
                    try:
                        res.update(metadata[i])
                    except IndexError:
                        raise ValueError(
                            "`metadatas` length and `embeddings` length must be same"
                        )

                faces.append(res)

        if not faces:
            return [], failed

        embedding = None
        embeddings = None

        idxs = [i["id"] for i in faces]
        if check_similar:
            res = self.check_similar(embeddings=[i["embedding"] for i in faces])
            for i, r in enumerate(res):
                if r:
                    print(
                        f"Similar face {r} already exists. If you want to add anyway, set `check_similar` to `False`."
                    )
                    failed.append(faces[i]["index"])
                    faces[i] = None

            res = None

        # remove None faces
        faces = [i for i in faces if i is not None]
        if not faces:
            return idxs, failed

        metadata = []
        added_img = False
        ids = [i["id"] for i in faces]
        for i in faces:
            data = {"id": i["id"]}
            if "img" in i:
                added_img = True
                img = img_to_cv2(i["img"])
                rect = i["rect"]
                img = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
                self.imgdb._add(
                    img_id=i["id"],
                    img=img,
                )

            for j in i:
                if j == "rect":
                    data["rect"] = i[j].to_json()

                elif j not in ["id", "embedding", "img"]:
                    data[j] = i[j]

            metadata.append(data)

        embeddings = [f["embedding"] for f in faces]
        if isinstance(embeddings[0], np.ndarray):
            embeddings = [e.tolist() for e in embeddings]

        faces = None

        self.db.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
        )

        if added_img:
            self.imgdb.conn.commit()

        print(f"Added {len(ids)} faces.")
        print(f"Failed to add {len(failed)} faces.")

        return idxs, failed

    def search(
        self, *, embedding=None, img=None, include=None, top_k=1
    ) -> list[FaceResults]:
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        sincludes, include = get_include(default="distances", include=include)

        result = self.db.query(
            embeddings=embedding,
            top_k=top_k,
            include=sincludes,
        )

        return self.db.parser(result, imgdb=self.imgdb, include=include)  # type: ignore

    def query(
        self,
        *,
        embedding=None,
        img=None,
        name=None,
        include=None,
        top_k=1,
        **search_params,
    ) -> Union[list[FaceResults], FaceResults]:
        params = {
            "embeddings": None,
            "top_k": top_k,
            "where": None,
            "include": None,
        }

        params["include"], include = get_include(default=None, include=include)

        params["embeddings"] = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            raise_error=False,
        )

        if name is not None:
            params["where"] = {"name": name}

        for key, value in search_params.items():
            if params["where"] is None:
                params["where"] = {}
            params["where"][key] = value

        if params["embeddings"]:
            result = self.db.query(
                **params,
            )
            return self.db.parser(result, imgdb=self.imgdb, include=include)  # type: ignore

        elif params["where"] is not None:
            ids = self.all(include=None)
            ids = [id["id"] for id in ids]

            if not params["include"]:
                params["include"] = ["metadatas"]

            result = self.db.get(
                ids=ids,
                where=params["where"],
                include=params["include"],
            )

            return self.db.parser(
                result, imgdb=self.imgdb, include=include, query=False
            )

        else:
            raise ValueError("Either embedding, img or name must be provided")

    def get(self, id, include=None):
        sincludes, include = get_include(default="metadatas", include=include)
        result = self.db.get(ids=[id], include=sincludes)
        return self.db.parser(result, imgdb=self.imgdb, include=include, query=False)

    def update(
        self, id, name=None, embedding=None, img=None, only_face=False, **metadata
    ):
        faces = []

        if isinstance(id, list):
            data = []
            if name is not None:
                assert len(id) == len(name)
                assert isinstance(name, list)

            if img is not None:
                assert len(id) == len(img)
                assert isinstance(img, list)

            if embedding is not None:
                assert len(id) == len(embedding)
                assert isinstance(embedding, list)
                assert isinstance(embedding[0], np.ndarray) or isinstance(
                    embedding[0], list
                )

                if isinstance(embedding[0], np.ndarray):
                    embedding = [i.tolist() for i in embedding]

            data = metadata.get("metadata", None)

            if data is not None:
                assert len(id) == len(data)
                assert isinstance(data, list)
                assert isinstance(data[0], dict)

            if data is None:
                data = [{} for _ in range(len(id))]

            for i in range(len(id)):
                if name is not None:
                    data[i]["name"] = name[i]

                faces.append(
                    {
                        "id": id[i],
                        "embedding": embedding[i] if embedding is not None else None,
                        "img": img[i] if img is not None else None,
                        "metadata": data[i] if data else None,
                    }
                )
        else:
            assert isinstance(id, str)
            if name is not None:
                metadata["name"] = name

            if embedding is not None:
                assert isinstance(embedding, np.ndarray) or isinstance(embedding, list)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

            faces.append(
                {
                    "id": id,
                    "embedding": embedding if embedding is not None else None,
                    "img": img if img is not None else None,
                    "metadata": metadata if metadata else None,
                }
            )

        ids = [i["id"] for i in faces]
        embeddings = [i["embedding"] for i in faces]
        imgs = [i["img"] for i in faces]
        metadata = [i["metadata"] for i in faces]

        for i, img in enumerate(imgs):
            if img is not None:
                if only_face:
                    rects = self.extract_faces(img)
                    rect = rects[0]
                    img = img_to_cv2(img)
                    img = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
                    metadata[i]["rect"] = rects[0].to_json()

                self.imgdb._add(
                    img_id=ids[i],
                    img=img,
                )

        self.db.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
        )

    def delete(self, id):
        self.db.delete(ids=id)

    def all(self, include=None) -> FaceResults:
        dincludes, include = get_include(default=None, include=include)
        result = self.db.all(include=dincludes)
        return self.db.parser(result, imgdb=self.imgdb, include=include, query=False)[0]  # type: ignore
