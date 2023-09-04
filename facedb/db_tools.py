from datetime import datetime
import sqlite3
from typing import Union
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import warnings
import pprint
import json


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

    def __str__(self):
        return self.to_json()

    def __len__(self):
        return 4

    def __repr__(self):
        return f"<Rect x={self.x} y={self.y} width={self.width} height={self.height}>"

    def __str__(self):
        return pprint.pformat(self)

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


def get_embeddings(
    imgs=None, embeddings=None, embedding_func=None, single=False, raise_error=True
):
    if single:
        if embeddings is None:
            if imgs is None:
                if raise_error:
                    raise ValueError("Either `embeddings` or `imgs` must be provided.")
                return None

            embeddings = embedding_func(imgs)
            if len(embeddings) > 1:
                warnings.warn(
                    "Multiple faces found in the img. If you are not sure use `search_many`. Taking first embedding."
                )

            if len(embeddings) == 0:
                raise ValueError("No face found in the img.")

        if isinstance(embeddings[0], np.ndarray) or isinstance(embeddings[0], list):
            embeddings = embeddings[0]

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings

    else:
        if embeddings is None:
            if imgs is None:
                if raise_error:
                    raise ValueError("Either `embeddings` or `imgs` must be provided.")
                return None
            
            
            elif not isinstance(imgs, list):
                imgs = [imgs]

            embeddings = []
            for img in imgs:
                embeddings.extend(embedding_func(img))

        if len(embeddings) == 0:
            return ValueError("No face found in the imgs.")

        if isinstance(embeddings[0], np.ndarray):
            embeddings = [e.tolist() for e in embeddings]

        return embeddings


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

    def add(self, *, img_id: Union[str, list], img: Union[str, list]):
        if isinstance(img, list):
            if len(img) != len(img_id):
                raise ValueError("Length of `img` and `img_id` must be same.")
            for i, j in zip(img_id, img):
                self._add(img_id=i, img=j)
        else:
            self._add(img_id=img_id, img=img)
        self.conn.commit()

    def add_rects(self, *, img, img_ids: list[str], rects: list[Rect], zoom_out=0.25):
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
        self.cursor.execute("""UPDATE img SET img=? WHERE img_id=?""", (img, img_id))
        self.conn.commit()


face_recognition_space_map = {
    "cosine": {
        "threshold": 0.63,
        "func": lambda dis, threshold: dis <= threshold,
    },
    "l2": {
        "threshold": 0.27,
        "func": lambda dis, threshold: dis <= threshold,
    },
    "ip": {
        "threshold": -1.4,
        "func": lambda dis, threshold: dis <= threshold,
    },
}

deeface_metric_map = {
    "cosine": "cosine",
    "euclidean": "l2",
    "euclidean_l2": "l2",
}

def face_recognition_is_similar(dis, threshold, space="cosine"):
    if threshold is None:
        threshold = face_recognition_space_map[space]["threshold"]

    return face_recognition_space_map[space]["func"](dis, threshold)


def time_now():
    return datetime.now().strftime("%m-%d-%Y-%I-%M-%S-%p")


class FaceResults(list["FaceResult"]):
    def __repr__(self):
        return f"<FaceResults count={len(self)}>"

    def __str__(self):
        return pprint.pformat([i.__str__() for i in self])

    def show_img(self, per_row=5, limit=10, page=1, img_size=(100, 100)):
        import matplotlib.pyplot as plt

        if not self or self[0].get("img", None) is None:
            print("No image available")
            return

        if len(self) > limit:
            self = self[limit * (page - 1) : limit * page]

        # resize
        for i in self:
            i["img"] = cv2.resize(i["img"], img_size)
            i["img"] = cv2.cvtColor(i["img"], cv2.COLOR_BGR2RGB)

        num_images = len(self)
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
                ax.imshow(self[i]["img"])
                ax.axis("off")
            else:
                ax.axis("off")

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

    @classmethod
    def from_query(
        cls,
        result,
        include: Union[list[str], str],
        imgdb: ImgDB,
        single=False,
    ) -> Union["FaceResult", "FaceResults"]:
        results: list[FaceResult] = []
        for r in range(len(result["ids"])):
            rs = []
            for i, idx in enumerate(result["ids"][r]):
                data = {"id": idx}
                for key in include:
                    if key[:9] == "embedding":
                        data["embedding"] = result["embeddings"][r][i]
                    elif key[:3] == "img":
                        data["img"] = imgdb.get(idx)
                    elif key[:8] == "distance":
                        data["distance"] = result["distances"][r][i]
                    else:
                        try:
                            data[key] = result["metadatas"][r][i][key]
                        except KeyError:
                            data[key] = None

                rs.append(FaceResult(**data))

            if len(rs) == 1:
                results.append(rs[0])
            elif len(rs) == 0:
                results.append(None)
            else:
                results.append(FaceResults(rs))

        return results

    @classmethod
    def from_get(
        cls,
        result,
        include,
        imgdb: ImgDB,
        single=False,
    ) -> Union["FaceResult", list["FaceResult"], None]:
        results = []
        for i, idx in enumerate(result["ids"]):
            data = {"id": idx}
            for key in include:
                if key[:9] == "embedding":
                    data["embedding"] = result["embeddings"][i]
                elif key[:3] == "img":
                    data["img"] = imgdb.get(idx)
                else:
                    try:
                        data[key] = result["metadatas"][i][key]
                    except KeyError:
                        data[key] = None

            results.append(FaceResult(**data))

        if single and results:
            return results[0]

        return FaceResults(results) or None
