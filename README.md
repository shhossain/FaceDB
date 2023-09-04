# FaceDB - A Face Recognition Database

FaceDB is a Python library that provides an easy-to-use interface for face recognition and face database management. It allows you to perform face recognition tasks, such as face matching and face searching, and manage a database of faces efficiently. FaceDB supports two popular face recognition frameworks: DeepFace and face_recognition.

## Features

- Face recognition using DeepFace or face_recognition.
- Efficient face database management.
- Face matching and searching.
- Support for multiple face recognition models and configurations.
- Easy integration into your Python projects.

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

```python
# Import the FaceDB library
from facedb import FaceDB

# Create a FaceDB instance
db = FaceDB()

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

## Documentation

Will be available soon.
