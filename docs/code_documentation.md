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

### FaceDB.add_many()

### FaceDB.recognize()

### FaceDB.all()

### FaceDB.all().df

### FaceDB.search()

### FaceDB.get_all()

### FaceDB.update()

### FaceDB.delete()

### Face.count()

### Face.query()

