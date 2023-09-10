# FaceDB - A Face Recognition Database

FaceDB is a Python library that provides an easy-to-use interface for face recognition and face database management. It allows you to perform face recognition tasks, such as face matching and face searching, and manage a database of faces efficiently. FaceDB supports two popular face recognition frameworks: DeepFace and face_recognition.

## Links
[Pypi](https://pypi.org/project/facedb/)
[Github](https://github.com/shhossain/facedb)

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

## Simple Usage

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

## Advanced Usage

You need to install pinecone first to use pinecone as the database backend.

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

# # Use can also use show_img() for multiple results
results = db.all(include='name')
results.show_img() # make sure you have matplotlib installed

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

# get all faces
faces = db.get_all(include=['name', 'img']) 

# Update a face
db.update(face_id, name="John Doe", img="john_doe.jpg", metadata1="metadata1", metadata2="metadata2")

# Delete a face
db.delete(face_id)

# Count the number of faces in the database
count = db.count()

# Get pandas dataframe of all faces
df = db.all().df
```

## Simple Querying

```python

# First add some faces to the database
db.add("Nelson Mandela", img="mandela.jpg", profession="Politician", country="South Africa")
db.add("Barack Obama", img="obama.jpg", profession="Politician", country="USA")
db.add("Einstein", img="einstein.jpg", profession="Scientist", country="Germany")

# Query the database by name
results = db.query(name="Nelson Mandela")

# Query the database by profession
results = db.query(profession="Politician")
```

## Advanced Querying

You can use following operators in queries:

- $eq - Equal to (number, string, boolean)
- $ne - Not equal to (number, string, boolean)
- $gt - Greater than (number)
- $lt - Less than (number)
- $in - In array (string or number)
- $regex - Regex match (string)

```python
results = db.query(
    profession={"$eq": "Politician"},
    country={"$in": ["USA", "South Africa"]},
)
# or write in a single json
results = db.query(
    where={
        "profession": {"$eq": "Politician"},
        "country": {"$in": ["USA", "South Africa"]},
    }
)

# you can use show_img(), df, query to further filter the results
results.show_img()
results.df
results.query(name="Nelson Mandela")

```
