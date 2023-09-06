# FaceDB - A Face Recognition Database

FaceDB is a Python library that provides an easy-to-use interface for face recognition and face database management. It allows you to perform face recognition tasks, such as face matching and face searching, and manage a database of faces efficiently. FaceDB supports two popular face recognition frameworks: DeepFace and face_recognition.

## Installation

FaceDB can be installed using pip:

```bash
pip install facedb
```

You can use face_recognition or DeepFace for face recognition. If you want to use DeepFace, you need to install the following dependencies:
for face_recognition:

```bash
pip install face_recognition
```

for DeepFace:

```bash
pip install deepface
```

## Usage

This will create a chromadb database in the current directory.

```python
# Import the FaceDB library
from facedb import FaceDB

# Create a FaceDB instance
db = FaceDB(
    path="facedata",
)

# Add a new face to the database
face_id = db.add("John Doe", img="john_doe.jpg")

# Recognize a face
result = db.recognize(img="new_face.jpg")

# Check if the recognized face is similar to the one in the database
if result and result["id"] == face_id:
    print("Recognized as John Doe")
else:
    print("Unknown face")
```

## Use with pinecone

You need to install pinecone first

```bash
pip install pinecone
```

```python
import os

os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"
os.environ["PINECONE_ENVIRONMENT"] = "YOUR_ENVIRONMENT_NAME"

db = FaceDB(
    path="facedata",
    metric='euclidean',
    database_backend='pinecone',
    index_name='faces',
    embedding_dim=128,
    module='face_recognition',
)

# This will create a pinecone index with name 'faces' in your environment if it doesn't exist

# add multiple faces
from glob import glob
from pathlib import Path

files = glob("faces/*.jpg") # Suppose you have a folder with imgs with names as filenames
imgs = []
names = []
for file in files:
    imgs.append(file)
    names.append(Path(file).name)

ids, failed_indexes = db.add_many(
    imgs=imgs,
    names=names,
)

unknown_face = "unknown_face.jpg"
result = db.recognize(img=unknown_face, include=['name'])
if result:
    print(f"Recognized as {result['name']}")
else:
    print("Unknown face")


# Include img in the result
result = db.recognize(img=unknown_face, include=['img'])
if result:
    result.show_img()

# or
img = result['img'] # cv2 image (numpy array)

# Include embedding in the result
result = db.recognize(img=unknown_face, include=['embedding'])
if result:
    print(result['embedding'])


# Search for similar faces
results = db.search(img=unknown_face, top_k=5, include=['name'])[0]

for result in results:
    print(f"Found {result['name']} with distance {result['distance']}")

# or search for multiple faces
multi_results = db.search(img=[img1, img2], top_k=5, include=['name'])

for results in multi_results:
    for result in results:
        print(f"Found {result['name']} with distance {result['distance']}")


```

## Documentation

Will be available soon.
