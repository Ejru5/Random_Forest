from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()


users = {
    1: {
        'id': 1,
        'name': 'John Doe',
        'city': 'Ahemdabad',
        'email': 'john.doea@gmail.com'

    },
    2: {
        'id': 2,
        'name': 'John Doe',
        'city': 'GGandhinagar',
        'email': 'janen.doea@gmail.com'

    },
    3: {
        'id': 3,
        'name': 'Jatan',
        'city': 'Jabalpur',
        'email': 'jatansahu2000@gmail.com'

    },
}

class User(BaseModel):
    id: int
    name: str
    city: str
    email: str


@app.get("/")  #http://localhost:8000 GET
def index():
    return {"message": "Hello World"}

@app.get("/test")
def index():
    return {"message": "Test API"}

@app.get("/users")
def get_users(city: str=None):   #It means city is optional
    if city is None:

        return { "message": "Users list", "data": list(users.values())}
    
    filtered_users = [user for user in users.values() if user.get('city').lower() == city.lower() ]
    return {'message': 'Users list', 'data': filtered_users}


@app.get('/users/{user_id}')   #GET baseUrl/users/1
def get_user_by_id(user_id: int ):
    return {'messages': 'user details' , 'data' : users[user_id] }

@app.post('/users')
def create_user(user: users):
    return {'messages': 'user details' , 'data' : user}
    pass
# @app.get('/users/{user_id}')
# def get_user_by_id(user_id: int , city: str=None):
#     return {'messages': 'user details' , 'data' : users[user_id] , 'filter': city}

# @app.get('/users/{user_id}')
# def get_user_by_id(user_id: int , city: str=None):
#     return {'messages': 'user details' , 'data' : users[user_id] , 'filter': city}

