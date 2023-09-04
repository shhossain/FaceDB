import chromadb
import numpy as np
from tqdm.auto import tqdm
from typing import Literal, Callable, Union
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
    face_recognition_is_similar,
    time_now,
    FaceResult,
    FaceResults,
    deeface_metric_map,
)
from pathlib import Path


def create_deepface_embedding_func(
    model_name,
    detector_backend,
    enforce_detection,
    align,
    normalization,
):
    def embedding_func(img, enforce_detection=enforce_detection, **kw):
        try:
            result = DeepFace.represent(
                img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                normalization=normalization,
            )
        except ValueError:
            return []

        return [i["embedding"] for i in result]

    return embedding_func


def create_face_recognition_embedding_func(
    model,
    num_jitters,
):
    def embedding_func(img, know_face_locations=None, **kw):
        img = img_to_cv2(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return face_recognition.face_encodings(
            img,
            num_jitters=num_jitters,
            model=model,
            known_face_locations=know_face_locations,
        )

    return embedding_func


def create_deepface_extract_faces_func(
    extract_faces_detector_backend,
    enforce_detection,
    align,
):
    def extract_faces(img):
        try:
            result = DeepFace.extract_faces(
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

        result = face_recognition.face_locations(img, model=extract_face_model)
        if result is None:
            return []

        rects = []
        for i in result:
            y, xw, yh, x = i
            w = xw - x
            h = yh - y
            rects.append(Rect(x, y, w, h))
        return rects

    return extract_faces


class FaceDB:
    def __init__(
        self,
        *,
        path=None,
        client=None,
        space: Literal["cosine", "l2", "ip"] = "cosine",
        embedding_func=None,
        module: Literal["deepface", "face_recognition"] = "deepface",
        **module_kwargs,
    ):
        if path is None:
            path = "data"
        if client is None:
            client = chromadb.PersistentClient(path)

        path = Path(path)

        self.imgdb = ImgDB(db_path=path / "img.db")

        self.client = client
        self.embedding_func: Callable[[np.ndarray], list[np.ndarray]] = None
        self.extract_faces: Callable[[np.ndarray], list[Rect]] = None
        self.module = module
        load_module(module)

        self.deepface_model_name = module_kwargs.get("model_name", "Facenet")
        self.space = space

        if embedding_func is None:
            if module == "deepface":
                self.embedding_func = create_deepface_embedding_func(
                    model_name=module_kwargs.pop("model_name", "Facenet"),
                    detector_backend=module_kwargs.pop("detector_backend", "ssd"),
                    enforce_detection=module_kwargs.pop("enforce_detection", True),
                    align=module_kwargs.pop("align", True),
                    normalization=module_kwargs.pop("normalization", "base"),
                )
                self.extract_faces = create_deepface_extract_faces_func(
                    extract_faces_detector_backend=module_kwargs.pop(
                        "extract_face_backend",
                        "ssd",
                    ),
                    enforce_detection=module_kwargs.pop("enforce_detection", True),
                    align=module_kwargs.pop("align", True),
                )
            elif module == "face_recognition":
                self.embedding_func = create_face_recognition_embedding_func(
                    model=module_kwargs.pop("model", "small"),
                    num_jitters=module_kwargs.pop("num_jitters", 1),
                )
                self.extract_faces = create_face_recognition_extract_faces_func(
                    extract_face_model=module_kwargs.pop("extract_face_model", "hog"),
                )

            else:
                raise ValueError(
                    "Currently only `deepface` and `face_recognition` are supported."
                )

        if space != "cosine":
            warnings.warn(
                "Only `cosine` space is tested. Other spaces may not work as expected."
            )

        self.db = client.get_or_create_collection(
            name="face_database",
            metadata={
                "hnsw:space": space,
            },
        )

    def __len__(self):
        return self.db.count()

    def count(self):
        return self.db.count()

    def _is_match(self, distance, threshold=None):
        if self.module == "deepface":
            threshold = deepface_distance.findThreshold(
                self.deepface_model_name, deeface_metric_map[self.space]
            )
            if distance <= threshold:
                return True
        else:
            if face_recognition_is_similar(distance, threshold, self.space):
                return True

        return False

    def get_face(self, img):
        rects = self.extract_faces(img)
        if rects:
            if len(rects) == 1:
                return img[
                    rects[0].y : rects[0].y + rects[0].h,
                    rects[0].x : rects[0].x + rects[0].w,
                ]
            else:
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
            query_embeddings=embeddings,
            n_results=1,
            include=["distances"],
        )
        results = FaceResult.from_query(result, include=["distance"], imgdb=self.imgdb)
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
        self, *, img=None, embedding=None, include=None, threshold=None, n_results=1
    ):
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            single=False,
        )

        if include is None:
            include = ["distance"]
        else:
            if "distance" not in include:
                include.append("distance")

        rincludes, include = get_include(default="distances", include=include)

        result = self.db.query(
            query_embeddings=embedding,
            n_results=n_results,
            include=rincludes,
        )

        results = FaceResult.from_query(result, include, single=False, imgdb=self.imgdb)
        res = []
        if results:
            for result in results:
                pr = []
                if isinstance(result, FaceResults):
                    for r in result:
                        if self._is_match(r["distance"]):
                            pr.append(r)
                    if len(pr) == 0:
                        res.append(None)
                    elif len(pr) == 1:
                        res.append(pr[0])
                    else:
                        res.append(FaceResults(pr))
                else:
                    if self._is_match(result["distance"]):
                        res.append(result)
                    else:
                        res.append(None)

        if len(res) == 1:
            return res[0]

        return res or None

    def add(
        self,
        name,
        embedding=None,
        img=None,
        id=None,
        check_similar=True,
        just_face=False,
        **metadata,
    ) -> str:
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            single=True,
        )

        if check_similar:
            result = self.check_similar(embeddings=[embedding])[0]
            if result:
                warnings.warn(
                    "Similar face already exists. If you want to add anyway, set `check_similar` to `False`."
                )
                return result

        metadata["name"] = name
        idx = id or name + "-" + time_now()
        if img is not None:
            if just_face:
                rects = self.extract_faces(img)
                rect = rects[0]
                img = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]

            self.imgdb.add(img_id=idx, img=img)

        self.db.add(
            ids=[idx],
            embeddings=[embedding],
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
    ) -> list[str]:
        faces = []
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
                rects = self.extract_faces(img)
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
                    metadata_posible = False
                    for j, embedding in enumerate(result):
                        name = f"{names[i]}_{j}"
                        faces.append(
                            {
                                "id": f"{idx}_{j}",
                                "name": name,
                                "embedding": embedding,
                                "img": img,
                                "rect": rects[j],
                            }
                        )
                elif len(result) > 0:
                    res = {
                        "id": idx,
                        "name": names[i],
                        "embedding": result[0],
                        "img": img,
                        "rect": rects[0],
                    }
                    if metadata_posible:
                        try:
                            res.update(metadata[i])
                        except IndexError:
                            pass
                    faces.append(res)
                else:
                    warnings.warn(f"No face found in the img {i}. Skipping.")

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
                        warnings.warn(
                            "Multiple faces found in the img. Taking first face."
                        )
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
            return []

        embedding = None
        embeddings = None

        idxs = [i["id"] for i in faces]
        if check_similar:
            res = self.check_similar(embeddings=[i["embedding"] for i in faces])
            for i, r in enumerate(res):
                if r:
                    warnings.warn(
                        f"Similar face {r} already exists. If you want to add anyway, set `check_similar` to `False`."
                    )
                    faces[i] = None
                    idxs[i] = r
            res = None

        # remove None faces
        faces = [i for i in faces if i is not None]
        if not faces:
            return idxs

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

        return idxs

    def search(
        self, *, embedding=None, img=None, include=None, n_results=1
    ) -> Union[list[FaceResult], FaceResult]:
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            single=False,
        )

        sincludes, include = get_include(default="distances", include=include)

        result = self.db.query(
            query_embeddings=embedding,
            n_results=n_results,
            include=sincludes,
        )

        return FaceResult.from_query(result, include, single=False, imgdb=self.imgdb)

    def query(
        self,
        *,
        embedding=None,
        img=None,
        name=None,
        include=None,
        n_results=1,
        **search_params,
    ) -> list[FaceResult]:
        params = {
            "query_embeddings": None,
            "n_results": n_results,
            "where": None,
            "include": None,
        }

        params["include"], include = get_include(default=None, include=include)

        params["query_embeddings"] = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            single=False,
            raise_error=False,
        )

        if name is not None:
            params["where"] = {"name": name}

        for key, value in search_params.items():
            if params["where"] is None:
                params["where"] = {}
            params["where"][key] = value

        if params["query_embeddings"]:
            result = self.db.query(
                **params,
            )
            return FaceResult.from_query(
                result, include, single=False, imgdb=self.imgdb
            )

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

            return FaceResult.from_get(result, include, single=False, imgdb=self.imgdb)

        else:
            raise ValueError("Either embedding, img or name must be provided")

    def get(self, id, include=None):
        sincludes, include = get_include(default="metadatas", include=include)

        result = self.db.get(ids=[id], include=sincludes)

        if result["ids"][0]:
            return FaceResult.from_get(result, include, single=True, imgdb=self.imgdb)

        return None

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

    def all(self, include=None):
        dincludes, include = get_include(default=None, include=include)
        result = self.db.get(include=dincludes)
        return FaceResult.from_get(result, include, single=False, imgdb=self.imgdb)
