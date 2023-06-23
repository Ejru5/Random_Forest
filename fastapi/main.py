from typing import Union
from fastapi import FastAPI,Header
from pydantic import BaseModel
from typing_extensions import Annotated
from sqlmodel import SQLModel,create_engine,Field,Session,update

app = FastAPI()
#postgressql://<username>:<password>@localhost/fastapi, localhost -> ip addrress/domain name
#create user username with encryption password 'password'
DATABASE_URL = 'postgresql://postgres:admin@localhost/fastapi'
engine = create_engine(DATABASE_URL)
def creat_db_tables():
    SQLModel.metadata.create_all(engine)

# class Users(SQLModel, table=True):
#     id : Optional[int] = Field(default=None,primary_key = True)
#     name : str
#     city : str
#     email : str

users = {
    1 : {'id':1,
        "name":"John",
        "city":"Gandhinagar",
        "email":"john@gmail.com"},
        
    2 : {'id':2,
        "name":"Emili",
        "city":"Surat",
        "email":"emili@gmail.com"},
    
    3 : {'id':3,
        "name":"Miya",
        "city":"Bombay",
        "email":"miya@gmail.com"}
}

class User(BaseModel):
    id : int
    name : str
    city : str
    email : str

class UserUpdate(BaseModel):
    name : str
    city : str
    email : str

@app.get("/") # @ - decorator , "get('/hello')"--> http://localhost:8000/hello
def index(): # index is arbitary
    return {"Message" : "Hello World"}

@app.get("/test") # @ - decorator , "get('/test')"--> http://localhost:8000/test
def test():
    a = 5
    return {"Message" : "Testing World",
            "Value" : a}

@app.get("/users")
def get_users(x_api_key: Annotated[Union[str,None], Header()], city:str=None): # Annotated|str
    if city is None or city == "":
        return {"message": "Users List", "data": users, "header" : x_api_key}
    else:
        list = [x for x in users.values() if x.get('city').lower()==city.lower()]
        return {"message": "Users List", "data": list, "filter":city, "header": x_api_key}


# @app.get("/users/city")
# def get_users_by_filter(city : str = None):
#     list = [x for x in users.values() if x.get('city').lower()==city.lower()]
#     return {"message": "Users List", "data": list, "filter":city}

@app.get("/users/{user_id}") # GET baseURL/users/1
def get_user_by_id(user_id:int):
    return {"message": "Users List", "data": users[user_id]}

@app.post("/users")
def create_user(user : User):
    # users[user.id] = user
    users.update({user.id: dict(user)})
    return {"message" : "New User Added", "Users": users}

@app.put("/users/{user_id}")
def update_user(user_id:int, user:UserUpdate):
    if user_id in users.keys():
        updated_user = users.get(user_id)
        updated_user.update({"city": user.city})  
        updated_user.update({"email": user.email})
        updated_user.update({"name": user.name})
        users.update({user_id : updated_user})
        return {"message" : "User Updated", "Users":users}
    else:
        return "User ID Doesn't Exist"
    
@app.delete("/users/{user_id}")
def delete_user(user_id:int):
    if user_id in users.keys():
        del users[user_id]
        return {"message":"User Deleted","Users":users}
    else:
        return {"message":"User_id is not valid"}