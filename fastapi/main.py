from typing import Union,Optional
from fastapi import FastAPI,Header,HTTPException
from pydantic import BaseModel
from typing_extensions import Annotated
from sqlmodel import SQLModel,create_engine,Field,Session,update

app = FastAPI()
#postgressql://<username>:<password>@localhost/fastapi, localhost -> ip addrress/domain name
#create user username with encryption password 'password'
# DATABASE_URL = 'postgresql://postgres:admin@localhost:5433/fastapi'
# engine = create_engine(DATABASE_URL)
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread":False})
# connect_args={"check_same_thread":False} only in sqlite not in postgres
def creat_db_tables():
    SQLModel.metadata.create_all(engine)

class Users(SQLModel, table=True):
    id : Optional[int] = Field(default=None,primary_key = True)
    name : str
    city : str
    email : str

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

@app.on_event("startup")
def on_startup():
    creat_db_tables()

@app.get("/") # @ - decorator , "get('/hello')"--> http://localhost:8000/hello
def index(): # index is arbitary
    return {"Message" : "Hello World"}

@app.get("/test") # @ - decorator , "get('/test')"--> http://localhost:8000/test
def test():
    a = 5
    return {"Message" : "Testing World",
            "Value" : a}

@app.get("/users", status_code=200)
def get_users(x_api_key: Annotated[Union[str,None], Header()], city:str=None): # Annotated|str
    # if city is None or city == "":
    #     return {"message": "Users List", "data": users, "header" : x_api_key}
    # else:
    #     list = [x for x in users.values() if x.get('city').lower()==city.lower()]
    #     return {"message": "Users List", "data": list, "filter":city, "header": x_api_key}
    with Session(engine) as session:
        users = session.query(Users).all() # select * from users
        print(f'users list {users}')
        return {"message": "Users list", "data":users,"header":x_api_key}


# @app.get("/users/city")
# def get_users_by_filter(city : str = None):
#     list = [x for x in users.values() if x.get('city').lower()==city.lower()]
#     return {"message": "Users List", "data": list, "filter":city}

@app.get("/users/{user_id}") # GET baseURL/users/1
def get_user_by_id(user_id:int):
    with Session(engine) as session:
        user = session.query(Users).filter(User.id == user_id).one_or_none() 
    # return {"message": "Users List", "data": users[user_id]}
        return {"message": "Users Details", "data": user}

@app.post("/users",status_code=201)
# def create_user(user : User):
#     # users[user.id] = user
#     users.update({user.id: dict(user)})
#     return {"message" : "New User Added", "Users": users}
def create_user(user:Users):
    with Session(engine) as session:
        session.add(user)
        session.commit()
        session.refresh(user)
        return {"message":"New User", "data": user}

@app.put("/users/{user_id}")
# def update_user(user_id:int, user:UserUpdate):
#     if user_id in users.keys():
#         updated_user = users.get(user_id)
#         updated_user.update({"city": user.city})  
#         updated_user.update({"email": user.email})
#         updated_user.update({"name": user.name})
#         users.update({user_id : updated_user})
#         return {"message" : "User Updated", "Users":users}
#     else:
#         return "User ID Doesn't Exist"
def update_user(user_id:int, user:Users):
    with Session(engine) as session:
        user_exist = session.query(Users).filter(Users.id == user_id).one_or_none()
        if not user_exist:
            raise HTTPException(404, 'Invalid User ID')
        
        user_exist.name = user.name
        user_exist.city = user.city
        user_exist.email = user.email
        session.add(user_exist)
        session.commit()
        session.refresh(user_exist)
        return {"message":"User Updated Successfully","data":user_exist}
    


@app.delete("/users/{user_id}")
# def delete_user(user_id:int):
#     if user_id in users.keys():
#         del users[user_id]
#         return {"message":"User Deleted","Users":users}
#     else:
#         return {"message":"User_id is not valid"}
def delete_user(user_id:int):
    with Session(engine) as session:
        user_exist = session.query(Users).filter(Users.id == user_id).one_or_none()
        if not user_exist:
            raise HTTPException(404, 'Invalid User ID')

        session.delete(user_exist)
        session.commit()
        return {"message":"User Delted Successfully", "data":user_exist}