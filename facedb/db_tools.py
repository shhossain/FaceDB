from datetime import datetime
import sqlite3

try:
    from typing import Optional, Union, List, Callable, Literal, Tuple
except ImportError:
    from typing_extensions import Optional, Union, List, Callable, Literal, Tuple

import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import warnings
import pprint
import json


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def is_none_or_empty(x):
    if x is None:
        return True
    else:
        return len(x) == 0


def img_to_cv2(img):
    if isinstance(img, str) or isinstance(img, Path):
        img = str(img)
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    elif isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, io.BytesIO):
        img = np.frombuffer(img.getvalue(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        raise TypeError(f"Unknown type of img: {type(img)}")
    return img


def img_to_bytes(img):
    if isinstance(img, str) or isinstance(img, Path):
        img = str(img)
        with open(img, "rb") as f:
            img = f.read()
    elif isinstance(img, np.ndarray):
        img = cv2.imencode(".jpg", img)[1].tobytes()
    elif isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.imencode(".jpg", img)[1].tobytes()
    elif isinstance(img, io.BytesIO):
        img = img.getvalue()
    else:
        raise TypeError(f"Unknown type of img: {type(img)}")
    return img


def many_vectors(obj):
    if isinstance(obj, list):
        if len(obj) == 0:
            return False
        elif isinstance(obj[0], list):
            return True
        elif isinstance(obj[0], np.ndarray):
            return True
    return False


def is_list_of_img(obj):
    if isinstance(obj, list):
        return True
    return False


def is_2d(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 2:
        return True
    return False


def convert_shape(x):
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 3:
        x = x.squeeze()
    return x


def get_embeddings(
    imgs: Optional[Union[str, List[str], np.ndarray, List[np.ndarray]]] = None,  # type: ignore
    embeddings: Optional[Union[List[List[float]], List[np.ndarray]]] = None,  # type: ignore
    embedding_func: Optional[
        Callable[[Union[str, np.ndarray]], Union[List[float], np.ndarray]]
    ] = None,
    raise_error: bool = True,
) -> List[List[float]]:
    if embeddings is None:
        if imgs is None:
            if raise_error:
                raise ValueError("imgs and embeddings cannot be both None")
            else:
                return []

        elif not is_list_of_img(imgs):
            imgs = [imgs]  # type: ignore

        if embedding_func is None:
            if raise_error:
                raise ValueError("embedding_func cannot be None")
            else:
                return []

        embeddings = []
        for img in imgs:  # type: ignore
            embeds = embedding_func(img)
            if is_none_or_empty(embeds):
                continue
            for embed in embeds:
                embeddings.append(embed)  # type: ignore

    embeddings = convert_shape(embeddings).tolist()

    return embeddings  # type: ignore


def get_include(default=None, include=None):
    if include is None:
        include = []
    elif isinstance(include, str):
        include = [include]

    sincludes = [default] if default else []
    if "embedding" in include:
        sincludes.append("embeddings")

    if include:
        sincludes.append("metadatas")

    return sincludes, include


face_recognation_metric_threshold = {
    "pinecone": {
        "cosine": {"value": 0.07, "operator": "le", "direction": "negative"},
        "cosine_l2": {
            "value": 0.07,
            "operator": "le",
            "direction": "negative",
        },
        "dotproduct": {
            "value": -0.8,
            "operator": "ge",
            "direction": "positive",
        },
        "dotproduct_l2": {
            "value": 0.07,
            "operator": "le",
            "direction": "negative",
        },
        "euclidean": {
            "value": 0.72,
            "operator": "ge",
            "direction": "positive",
        },
        "euclidean_l2": {
            "value": 0.85,
            "operator": "ge",
            "direction": "positive",
        },
    },
    "chromadb": {
        "cosine": {"value": 0.06, "operator": "le", "direction": "negative"},
        "cosine_l2": {
            "value": 0.07,
            "operator": "le",
            "direction": "negative",
        },
        "ip": {"value": -1.1, "operator": "ge", "direction": "positive"},
        "ip_l2": {"value": 0.07, "operator": "le", "direction": "negative"},
        "l2": {"value": 0.27, "operator": "le", "direction": "negative"},
        "l2_l2": {"value": 0.14, "operator": "le", "direction": "negative"},
    },
}


def face_recognition_is_match(
    db_backend, metric, value, l2_normalization=True, threshold: Optional[float] = None
):
    if l2_normalization:
        metric_threshold = face_recognation_metric_threshold[db_backend][metric + "_l2"]
    else:
        metric_threshold = face_recognation_metric_threshold[db_backend][metric]

    if threshold is None:
        threshold = metric_threshold["value"]

    if metric_threshold["operator"] == "le":
        return value <= threshold
    elif metric_threshold["operator"] == "ge":
        return value >= threshold
    else:
        raise ValueError(f"Unknown operator: {metric_threshold['operator']}")


deeface_metric_map = {
    "cosine": "cosine",
    "euclidean": "l2",
    "euclidean_l2": "l2",
}


def time_now():
    return datetime.now().strftime("%m-%d-%Y-%I-%M-%S-%p")


def get_model_dimension(module, model_name):
    if module == "face_recognition":
        return 128
    elif module == "deepface":
        dim_map = {
            "VGG-Face": 2622,
            "Facenet": 128,
            "Facenet512": 512,
            "OpenFace": 128,
            "DeepFace": 8631,
            "DeepID": 160,
            "Dlib": 128,
            "ArcFace": 512,
            "Ensemble": 8631,
        }
        return dim_map[model_name]
    else:
        raise ValueError(f"Unknown module: {module}")


class Rect(dict):
    def __init__(self, x, y, w, h):
        super().__init__(x=x, y=y, width=w, height=h)

        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def width(self):
        return self.w

    @width.setter
    def width(self, value):
        self.w = value

    @property
    def height(self):
        return self.h

    @height.setter
    def height(self, value):
        self.h = value

    @classmethod
    def from_json(cls, json):
        if "width" in json:
            return cls(json["x"], json["y"], json["width"], json["height"])
        else:
            return cls(json["x"], json["y"], json["w"], json["h"])

    def to(self, module):
        if module == "face_recognition":
            return (self.y, self.x + self.w, self.y + self.h, self.x)
        else:
            return [self.x, self.y, self.w, self.h]

    def to_json(self):
        return json.dumps(self)

    def __len__(self):
        return 4

    def __repr__(self):
        return f"<Rect x={self.x} y={self.y} width={self.width} height={self.height}>"

    def __str__(self):
        return pprint.pformat(self.to_json())

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.width
        elif key == 3:
            return self.height
        else:
            raise IndexError("Rect index out of range")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.width = value
        elif key == 3:
            self.height = value
        else:
            raise IndexError("Rect index out of range")

    def __iter__(self):
        return iter([self.x, self.y, self.width, self.height])

    def __eq__(self, other):
        if isinstance(other, Rect):
            return all(
                [
                    self.x == other.x,
                    self.y == other.y,
                    self.width == other.width,
                    self.height == other.height,
                ]
            )
        elif isinstance(other, list) or isinstance(other, tuple):
            return all(
                [
                    self.x == other[0],
                    self.y == other[1],
                    self.width == other[2],
                    self.height == other[3],
                ]
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class ImgDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def __del__(self):
        self.conn.close()

    # img, id (str)
    def create_table(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS img (
            img_id TEXT PRIMARY KEY,
            img BLOB
        )"""
        )
        self.conn.commit()

    def _add(self, *, img_id, img):
        img = img_to_bytes(img)
        self.cursor.execute("""INSERT INTO img VALUES (?, ?)""", (img_id, img))

    def add(self, *, img_id: Union[str, list], img):
        if isinstance(img, list):
            if len(img) != len(img_id):
                raise ValueError("Length of `img` and `img_id` must be same.")
            for i, j in zip(img_id, img):
                self._add(img_id=i, img=j)
        else:
            self._add(img_id=img_id, img=img)
        self.conn.commit()

    def add_rects(self, *, img, img_ids: List[str], rects: List[Rect], zoom_out=0.25):
        img = img_to_cv2(img)
        img_h, img_w = img.shape[:2]
        for rect in rects:
            rect.x = max(0, rect.x - int(rect.w * zoom_out))
            rect.y = max(0, rect.y - int(rect.h * zoom_out))
            rect.w = min(img_w - rect.x, rect.w + int(rect.w * zoom_out))
            rect.h = min(img_h - rect.y, rect.h + int(rect.h * zoom_out))

        if len(img_ids) != len(rects):
            raise ValueError("Length of `img_ids` and `rects` must be same.")

        for img_id, rect in zip(img_ids, rects):
            self._add(
                img_id=img_id,
                img=img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w],
            )
        self.conn.commit()

    def get(self, img_id):
        self.cursor.execute("""SELECT img FROM img WHERE img_id=?""", (img_id,))
        img = self.cursor.fetchone()
        if img is None:
            return None

        try:
            img = cv2.imdecode(np.frombuffer(img[0], np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            warnings.warn(f"Error in decoding image: {e}")
            return None

        return img

    def delete(self, img_id):
        self.cursor.execute("""DELETE FROM img WHERE img_id=?""", (img_id,))
        self.conn.commit()

    def update(self, *, img_id, img):
        img = img_to_bytes(img)
        self.cursor.execute("""UPDATE img SET img=? WHERE img_id=?""", (img, img_id))
        self.conn.commit()

    def auto(self, *, img_id, img):
        img = img_to_bytes(img)
        self.cursor.execute(
            """INSERT OR REPLACE INTO img VALUES (?, ?)""", (img_id, img)
        )
        self.conn.commit()
