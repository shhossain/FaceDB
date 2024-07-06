import re
import numpy as np
from tqdm.auto import tqdm
import cv2
import warnings
import threading
import os

DeepFace = None
deepface_distance = None
face_recognition = None

from facedb.db_tools import (
    get_embeddings,
    get_include,
    ImgDB,
    Rect,
    img_to_cv2,
    is_list_of_img,
    is_2d,
    time_now,
    get_model_dimension,
    l2_normalize,
    is_none_or_empty,
    Union,
    Literal,
    Optional,
    List,
    Tuple,
    fthresholds,
    FailedImageIndexList,
)

from facedb.db_models import FaceResults, PineconeDB, ChromaDB

from pathlib import Path

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
            return None

        result = [i["embedding"] for i in result]
        if l2_normalization:
            result = l2_normalize(result)

        if is_none_or_empty(result):
            return None

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
        if is_none_or_empty(result):
            return None

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
        path: str = "facedata",
        metric: Literal["cosine", "euclidean", "dot"] = "euclidean",
        embedding_func=None,
        embedding_dim: Optional[int] = None,
        l2_normalization: bool = True,
        module: Literal["deepface", "face_recognition"] = "face_recognition",
        database_backend: Literal["chromadb", "pinecone"] = "chromadb",
        pinecone_settings: dict = {},
        face_recognition_settings: dict = {},
        deepface_settings: dict = {},
        **kw,
    ):
        """
        Initialize the FaceDB instance for face recognition.

        Args:
            path (str, optional): The path to store data. Defaults to "facedata".
            metric (str, optional): The distance metric to use for similarity. Defaults to "euclidean".
            embedding_func (callable, optional): Custom embedding function. Defaults to None.
            embedding_dim (int, optional): The dimension of face embeddings. Defaults to selected automatically.
            l2_normalization (bool, optional): Whether to perform L2 normalization on embeddings(increases accuracy). Defaults to True.
            module (str, optional): The face recognition module to use. Defaults to "face_recognition" (DeepFace not optimized).
            database_backend (str, optional): The database backend to use(ChromaDB or Pinecone). Defaults to "chromadb".

            pinecone_settings (dict, optional):
                Additional settings to pass to the Pinecone client.
                client (PineconeClient, optional): The Pinecone client to use. Defaults to HTTPClient.
                index_name (str, optional): The name of the Pinecone index to use.
                api_key (str, optional): Can be passed as `pinecone_api_key` or environment variable `PINECONE_API_KEY`.
                spec (str, optional): The Pinecone spec to use. Defaults to ServerlessSpec(cloud="aws",region="us-east-1").
                Rest of the keyword arguments are passed to the pinecone client directly.

            face_recognition_settings (dict, optional):
                Additional settings to pass to the face recognition module.
                model (str, optional): Model size. Defaults to "small".
                num_jitters (int, optional): Number of jitter samples. Defaults to 1.
                extract_face_model (str, optional): Face detection model. Defaults to "hog".

            deepface_settings (dict, optional):
                Additional settings to pass to the DeepFace module.
                model_name (str, optional): Model name. Defaults to "Facenet512".
                detector_backend (str, optional): Face detection backend. Defaults to "ssd".
                enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
                normalization (bool, optional): Whether to normalize face embeddings. Defaults to True.
                extract_face_backend (str, optional): Face detection backend. Defaults to "ssd".
                enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
                align (bool, optional): Whether to align faces. Defaults to True.



        Examples:
            >>> from facedb import FaceDB
            >>> facedb = FaceDB()
            >>> facedb.add("elon_musk", img="elon_musk.jpg")
            >>> facedb.add("jeff_bezos", img="jeff_bezos.jpg")
            >>> facedb.recognize(img="elon_musk_2.jpg") # returns FaceResults
        """

        path = Path(path)

        assert metric in [
            "cosine",
            "euclidean",
            "dot",
        ], "Supported metrics are `cosine`, `euclidean` and `dot`."
        assert module in [
            "deepface",
            "face_recognition",
        ], "Supported modules are `deepface` and `face_recognition`."
        assert database_backend in [
            "chromadb",
            "pinecone",
        ], "Supported database backends are `chromadb` and `pinecone`."

        if module == "deepface":
            warnings.warn(
                "Deepface module is not calibrated for vector database. Use `face_recognition` instead."
            )

        os.environ["DB_BACKEND"] = database_backend

        self.metric = metric
        self.embedding_func: Callable = embedding_func  # type: ignore
        self.extract_faces: Callable = None  # type: ignore
        self.module = module
        self.db_backend = database_backend
        self.l2_normalization = l2_normalization
        self.deepface_model_name = kw.get("model_name", "Facenet")

        if database_backend == "chromadb":
            self.db = ChromaDB(
                path=str(path),
                client=kw.pop("client", None),
                metric=metric_map[database_backend][metric],
                collection_name=kw.pop("collection_name", "facedb"),
            )

        elif database_backend == "pinecone":
            self.db = PineconeDB(
                pinecone_client=pinecone_settings.pop("client", None),
                index_name=pinecone_settings.pop("index_name", None),
                metric=metric_map[database_backend][metric],
                dimension=embedding_dim
                or get_model_dimension(module, self.deepface_model_name),
                api_key=pinecone_settings.pop("pinecone_api_key", None),
                spec=pinecone_settings.pop("spec", None),
                **pinecone_settings,
            )

        if embedding_func is None:
            load_module(module)
            if module == "deepface":
                self.embedding_func = create_deepface_embedding_func(
                    model_name=deepface_settings.pop("model_name", "Facenet"),
                    detector_backend=deepface_settings.pop("detector_backend", "ssd"),
                    enforce_detection=deepface_settings.pop("enforce_detection", True),
                    align=deepface_settings.pop("align", True),
                    normalization=deepface_settings.pop("normalization", "base"),
                    l2_normalization=l2_normalization,
                )
                self.extract_faces = create_deepface_extract_faces_func(
                    extract_faces_detector_backend=deepface_settings.pop(
                        "extract_face_backend",
                        "ssd",
                    ),
                    enforce_detection=deepface_settings.pop("enforce_detection", True),
                    align=deepface_settings.pop("align", True),
                )
            elif module == "face_recognition":
                self.embedding_func = create_face_recognition_embedding_func(
                    model=face_recognition_settings.pop("model", "small"),
                    num_jitters=face_recognition_settings.pop("num_jitters", 1),
                    l2_normalization=l2_normalization,
                )
                self.extract_faces = create_face_recognition_extract_faces_func(
                    extract_face_model=face_recognition_settings.pop(
                        "extract_face_model", "hog"
                    ),
                )
            else:
                raise ValueError(
                    "Currently only `deepface` and `face_recognition` are supported."
                )
        else:
            self.embedding_func = embedding_func

        if not path.exists():
            path.mkdir(parents=True)

        self.imgdb = ImgDB(db_path=str(path / "img.db"))
        self.threshold = self.get_threshold()

    def __len__(self):
        return self.db.count()

    def count(self):
        """
        Get the number of faces in the database (alias for __len__).

        Returns:
            int: The number of faces in the database.
        """
        return self.db.count()

    def get_threshold(self) -> Tuple[str, str, float]:
        """
        Get the similarity threshold for the database.

        Returns:
            Tuple[str, str, float]: The similarity threshold.
        """

        metric = self.metric
        if self.module == "deepface":
            if metric == "euclidean" and self.l2_normalization:
                metric = "euclidean_l2"

            threshold = deepface_distance.findThreshold(  # type: ignore
                self.deepface_model_name, metric
            )
            return "le", "negative", threshold

        elif self.module == "face_recognition":
            metric = metric_map[self.db_backend][metric]
            if self.l2_normalization:
                metric_threshold = fthresholds[self.db_backend][metric + "_l2"]
            else:
                metric_threshold = fthresholds[self.db_backend][metric]

            return (
                metric_threshold["operator"],
                metric_threshold["direction"],
                metric_threshold["value"],
            )

        else:
            raise ValueError(
                "Currently only `deepface` and `face_recognition` are supported."
            )

    def _is_match(self, distance, threshold=None):
        if threshold is None or threshold == 80:
            op, _, threshold = self.threshold
        else:
            op, _, thrs = self.threshold
            threshold = max(10, threshold)
            thrs = thrs / (threshold / 100)
            threshold = thrs

        if op == "le":
            return distance <= threshold
        elif op == "ge":
            return distance >= threshold
        elif op == "eq":
            return distance == threshold
        elif op == "l":
            return distance < threshold
        elif op == "g":
            return distance > threshold
        elif op == "ne":
            return distance != threshold
        else:
            raise ValueError("Invalid operator.")

    def get_faces(self, img, *, zoom_out=0.25, only_rect=False) -> Union[None, list]:
        """
        Extract faces from an image.

        Args:
            img: The input image.
            zoom_out (float, optional): Zoom factor for the extracted faces. Defaults to 0.25.
            only_rect (bool, optional): Whether to return only the face rectangles. Defaults to False.

        Returns:
            Union[None, list]: A list of extracted faces or face rectangles.
        """
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
        """
        Check for similar faces in the database.

        Args:
            embeddings: Face embeddings to compare.
            threshold (float, optional): The similarity threshold. Defaults to 80.

        Returns:
            list: List of id(if it is match) or false
        """
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
        """
        Recognize a face from an image or embedding.

        Args:
            img: The input image.
            embedding: Face embeddings for recognition.
            include (list, optional): List of information to include in the result. Defaults to None.
            threshold (float, optional): The similarity threshold. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 1.

        Returns:
            False: If no face is found in the image.
            None: If no match is found.
            FaceResults: If a match is found.
        """

        single = False
        size = 0
        if embedding is not None:
            if not is_2d(embedding):
                single = True
            else:
                size = len(embedding)
        elif img is not None:
            if not is_list_of_img(img):
                single = True
            else:
                size = len(img)

        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        if is_none_or_empty(embedding):
            if size == 0:
                return False
            return [False] * size

        rincludes, include = get_include(default="distances", include=include)
        result = self.db.query(
            embeddings=embedding,
            top_k=top_k,
            include=rincludes,
        )

        results = self.db.parser(result, imgdb=self.imgdb, include=include, threshold=self.threshold)  # type: ignore
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
        """
        Add a new face to the database.

        Args:
            name (str): The name of the person associated with the face.
            img: The input image.
            embedding: Face embeddings for the new face.
            id (str, optional): The unique ID for the face. Defaults to None.
            check_similar (bool, optional): Whether to check for similar faces. Defaults to True.
            save_just_face (bool, optional): Whether to save only the face region. Defaults to False.
            **metadata: Additional metadata for the face.

        Returns:
            str: The ID of the added face.

        Raises:
            ValueError: If no face is found in the image.
        """
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        if is_none_or_empty(embedding):
            raise ValueError("No face found in the img.")

        if check_similar:
            result = self.check_similar(embeddings=embedding)[0]
            if result:
                print(
                    "Similar face already exists. If you want to add anyway, set `check_similar` to `False`."
                )
                return result

        metadata["name"] = name
        idx = id or re.sub(r"[\W]", "-", name) + "-" + time_now()
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
    ) -> Tuple[list, list]:
        """
        Add multiple faces to the database.

        Args:
            embeddings: List of face embeddings to add.
            imgs: List of input images containing faces.
            metadata: List of metadata for the faces.
            ids (list, optional): List of unique IDs for the faces. Defaults to None.
            names (list, optional): List of names associated with the faces. Defaults to None.
            check_similar (bool, optional): Whether to check for similar faces. Defaults to True.

        Returns:
            tuple: A tuple containing lists of added IDs and failed IDs.
        """
        faces = []
        failed = FailedImageIndexList()
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
                cnames = [re.sub(r"[\W]", "-", name) for name in names]
                idxs = ids or [f"{name}-{time_now()}" for name in cnames]
            else:
                names = [f"face_{i}-{time_now()}" for i in range(len(imgs))]
                idxs = ids or [f"faceid_{i}-{time_now()}" for i in range(len(imgs))]

            for i, img in enumerate(tqdm(imgs, desc="Extracting faces")):
                try:
                    rects: List[Rect] = self.get_faces(img, only_rect=True)  # type: ignore
                    if is_none_or_empty(rects):
                        print(f"No face found in the img {i}. Skipping.")
                        failed.append(i, failed_reason="No face found in the img")
                        continue

                    result = self.embedding_func(
                        img,
                        know_face_locations=[r.to(self.module) for r in rects],
                        enforce_detection=False,
                    )

                    if is_none_or_empty(result):
                        print(f"No face found in the img {i}. Skipping.")
                        failed.append(i, failed_reason="No face found in the img")
                        continue

                    try:
                        idx = idxs[i]
                    except IndexError:
                        raise IndexError("`ids` length and `imgs` length must be same")

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
                        print(f"No face found in the img {i}. Skipping.")
                        failed.append(i, failed_reason="No face found in the img")
                        continue
                except Exception as e:
                    print(f"Error in img {i}. Skipping.", e)
                    failed.append(i, failed_reason=str(e))
                    continue

            result = None
        else:
            idxs = ids or [f"faceid_{i}-{time_now()}" for i in range(len(embeddings))]
            for i, embedding in enumerate(embeddings):
                try:
                    idx = idxs[i]
                except IndexError:
                    raise IndexError(
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
                        raise IndexError(
                            "`names` length and `embeddings` length must be same"
                        )
                else:
                    res["name"] = f"face_{i}-{time_now()}"

                if imgs is not None:
                    try:
                        rects = self.extract_faces(imgs[i])
                    except IndexError:
                        raise IndexError(
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
                        raise IndexError(
                            "`metadatas` length and `embeddings` length must be same"
                        )

                faces.append(res)

        if not faces:
            return [], failed

        embedding = None
        embeddings = None

        if check_similar:
            res = self.check_similar(embeddings=[i["embedding"] for i in faces])
            for i, r in enumerate(res):
                if r:
                    print(
                        f"Similar face {r} already exists. If you want to add anyway, set `check_similar` to `False`."
                    )
                    failed.append(
                        faces[i]["index"],
                        failed_reason=f"Similar face {r} already exists.",
                    )
                    faces[i] = None

            res = None

        # remove None faces
        faces = [i for i in faces if i is not None]
        if not faces:
            return [], failed

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

        return ids, failed

    def search(
        self, *, embedding=None, img=None, include=None, top_k=1
    ) -> List[FaceResults]:
        """
        Search for similar faces in the database.

        Args:
            embedding: Face embeddings for searching.
            img: The input image for searching.
            include (list, optional): List of information to include in the result. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 1.

        Returns:
            list: List of search results.
        """
        embedding = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
        )

        if is_none_or_empty(embedding):
            return []

        sincludes, include = get_include(default="distances", include=include)

        result = self.db.query(
            embeddings=embedding,
            top_k=top_k,
            include=sincludes,
        )

        return self.db.parser(result, imgdb=self.imgdb, include=include, threshold=self.threshold)  # type: ignore

    def query(
        self,
        *,
        embedding=None,
        img=None,
        name=None,
        include=None,
        top_k=1,
        **search_params,
    ) -> Union[List[FaceResults], FaceResults]:
        """
        Query the database for faces based on specified parameters.

        Args:
            embedding: Face embeddings for querying.
            img: The input image for querying.
            name (str, optional): The name associated with the face. Defaults to None.
            include (list, optional): List of information to include in the result. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 1.
            **search_params: Additional search parameters.

        Returns:
            Union[List[FaceResults], FaceResults]: Query results.
        """
        params = {
            "embeddings": None,
            "top_k": top_k,
            "where": {},
            "include": None,
        }

        params["include"], include = get_include(default=None, include=include)

        params["embeddings"] = get_embeddings(
            embeddings=embedding,
            imgs=img,
            embedding_func=self.embedding_func,
            raise_error=False,
        )

        if not params["embeddings"]:
            params["embeddings"] = None

        if name is not None:
            params["where"]["name"] = name

        if search_params.get("where", None) is None:
            for key, value in search_params.items():
                params["where"][key] = value
        else:
            params["where"] = search_params["where"]

        if not params["where"]:
            params["where"] = None

        if params["embeddings"]:
            result = self.db.query(
                **params,
            )
            return self.db.parser(result, imgdb=self.imgdb, include=include, threshold=self.threshold)  # type: ignore

        elif params["where"] is not None:
            return self.all(include=["name"]).query(**params["where"])
        else:
            raise ValueError("Either embedding, img or name must be provided")

    def get(self, id, include=None):
        """
        Retrieve information about a specific face from the database.

        Args:
            id (str): The ID of the face to retrieve.
            include (list, optional): List of information to include in the result. Defaults to None.

        Returns:
            FaceResults: Information about the retrieved face.
        """
        sincludes, include = get_include(default="metadatas", include=include)
        result = self.db.get(ids=id, include=sincludes)
        return self.db.parser(result, imgdb=self.imgdb, include=include, query=False)

    def update(
        self, id, name=None, embedding=None, img=None, only_face=False, **metadata
    ):
        """
        Update information for a specific face in the database.

        Args:
            id (str): The ID of the face to update.
            name (str, optional): The new name associated with the face. Defaults to None.
            embedding: New face embeddings for the face.
            img: New input image for the face.
            only_face (bool, optional): Whether to update only the face region. Defaults to False.
            **metadata: Additional metadata to update.

        Returns:
            None

        Raises:
            ValueError: If id is not found.
        """
        result = self.get(id=id, include=["name"])
        if not result:
            raise ValueError(f"Face with id {id} not found.")

        data = {}
        data_update = False
        if name is not None:
            data["name"] = name
            data_update = True
        else:
            data["name"] = result["name"]  # type: ignore

        if metadata:
            data_update = True
            for key, value in metadata.items():
                data[key] = value

        if not data_update:
            data = None

        if img is not None:
            if only_face:
                img = self.get_faces(img)
                if img is None:
                    raise ValueError("No face found in the img.")
                img = img[0]

            self.imgdb.auto(img_id=id, img=img)

        self.db.update(
            ids=id,
            embeddings=embedding,
            metadatas=data,
        )

    def delete(self, id):
        """
        Delete a face from the database.

        Args:
            id (str) or ids (list) of face id.

        Returns:
            None
        """

        self.imgdb.delete(id)
        self.db.delete(ids=id)

    def delete_all(self):
        """
        Delete all faces from the database.  This action is irreversible. Use with caution.
        """
        self.db.delete_all()
        self.imgdb.delete_all()

    def all(self, include=None) -> FaceResults:
        """
        Retrieve information about all faces in the database.

        Args:
            include (list, optional): List of information to include in the result. Defaults to None.

        Returns:
            FaceResults: Information about all faces in the database.
        """
        dincludes, include = get_include(default=None, include=include)
        result = self.db.all(include=dincludes)
        return self.db.parser(result, imgdb=self.imgdb, include=include, query=False)  # type: ignore
