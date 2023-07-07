from fastapi import FastAPI
from pydantic import BaseModel
<<<<<<< Updated upstream
=======
from typing_extensions import Annotated
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
# from sqlmodel import SQLModel,create_engine,Field,Session,update


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers.core import Dense,Activation,Dropout
# from keras.layers import Dense, Activation, Dropout

# from keras.layers import LSTM
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
# from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from keras.callbacks import EarlyStopping
# import math
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        return { "message": "Users list", "data": list(users.values())}
=======
@app.post("/login")
async def login(request: Request):
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")

    # Perform authentication logic here (e.g., calling an authentication API)
    # Replace the following conditional statement with your authentication logic
    if username == "admin" and password == "password":
        return RedirectResponse(url="/home")
    else:
        return RedirectResponse(url="/")

@app.post("/home")
async def home_page():
    with open("home.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/page1")
async def page1():
    with open("page1.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/page2")
async def page2():
    with open("page2.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/page3")
async def page3():
    with open("page3.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)



@app.post('/city_based_swagger')
def city_based_recommendation(data:cityBased):
    hotelsswagger =  citybased(data.city)
    return {"request": data.city, "hotels": hotelsswagger}

@app.post('/city_based_recommendation')
async def city_based_recommendation(request: Request):
    form_data = await request.form()
    city = form_data.get("city")
    hotels   = citybased(city)

    return templates.TemplateResponse(
        "city_based.html",
        {"request": request, "hotels": hotels}
    )

    # if hotels == 'No Hotels Available':
    #     return JSONResponse(
    #         status_code=404,
    #         content={
    #             "message": "No hotel recommendations found for the given city."
    #         }
    #     )
    # else:
    #     return JSONResponse(
    #             status_code=200,
    #             content={
    #                 "message": "Hotel recommendations retrieved successfully.",
    #                 "hotels": hotels
    #             }
    #         )
>>>>>>> Stashed changes
    
    filtered_users = [user for user in users.values() if user.get('city').lower() == city.lower() ]
    return {'message': 'Users list', 'data': filtered_users}

@app.post('/requirement_based_swagger')
def requirement_based_recommendation(data:requirementBased):
    requireSwagger = requirementbased(data.city, data.number, data.features)
    return {"request": {"city":data.city, "peoples": data.number, "features":data.features }, "hotels": requireSwagger}

<<<<<<< Updated upstream
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

=======
@app.post("/requirement_based_recommendation")
async def requirement_based_recommendation(request: Request):
    form_data = await request.form()
    city = form_data.get("city")
    number = form_data.get("number")
    features = form_data.get("features")
    requireBased =  requirementbased(city, number, features)



    
    return templates.TemplateResponse(
        "feature_based.html",
        {"request": request, "hotels": requireBased}
    )



# @app.get("/city_based.html")
# async def city_based():
#     with open("city_based.html") as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content, status_code=200)

@app.get("/feature_based.html")
async def recommendation_page():
    with open("feature_based.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
>>>>>>> Stashed changes
