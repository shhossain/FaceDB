# FaceDB object initialization

# Main features

### FaceDB.add(name:str,img=None,embedding=None,id=None,check_similar=True,save_just_face=False,**kwargs: Additional metadata for the face.)
Give you the possibility to add a new entry in our FaceDB database.  
Example : 
```
db.add("Nelson Mandela", img="mandela.jpg", profession="Politician", country="South Africa")
db.add("Barack Obama", img="obama.jpg", profession="Politician", country="USA")
db.add("Einstein", img="einstein.jpg", profession="Scientist", country="Germany")
```

### FaceDB.add_many(embeddings=None,imgs=None,metadata=None,ids=None,names=None,check_similar=True)
Give you the possibility to add several new entries in our FaceDB database at one time.   
Example :
```
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
```

### FaceDB.recognize(img=None, embedding=None, include=None, threshold=None, top_k=1) -> False|None|FaceResults
Try to find the name of the personne within the picture.   
Example:
```
result = db.recognize(img="your_image.jpg")
```
### FaceDB.all(include=None) -> FaceResults
Retrieve information about all faces in the database.   
Example:
```
results = db.all(include='name')
#Or with a list
results = db.all(include=['name', 'img']) 
```
### FaceDB.all().df

### FaceDB.search()

### FaceDB.get_all()

### FaceDB.update()

### FaceDB.delete()

### Face.count()

### Face.query()

