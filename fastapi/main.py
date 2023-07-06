from typing import Union,Optional, Literal 
from fastapi import FastAPI,Header,HTTPException, Depends
from pydantic import BaseModel
from typing_extensions import Annotated
# from sqlmodel import SQLModel,create_engine,Field,Session,update


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
# from keras.layers.core import Dense,Activation,Dropout
from keras.layers import Dense, Activation, Dropout

from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import EarlyStopping
import math
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


app = FastAPI()

hotel_details=pd.read_csv('./data/Hotel_details.csv',delimiter=',')
hotel_rooms=pd.read_csv('./data/Hotel_Room_attributes.csv',delimiter=',')
hotel_cost=pd.read_csv('./data/hotels_RoomPrice.csv',delimiter=',')


del hotel_details['id']
del hotel_rooms['id']
del hotel_details['zipcode']

hotel_details=hotel_details.dropna()
hotel_rooms=hotel_rooms.dropna()

hotel_details.drop_duplicates(subset='hotelid',keep=False,inplace=True)

hotel=pd.merge(hotel_rooms,hotel_details,left_on='hotelcode',right_on='hotelid',how='inner')

del hotel['hotelid']
del hotel['url']
del hotel['curr']
del hotel['Source']

def citybased(city):
    hotel['city']=hotel['city'].str.lower()
    citybase=hotel[hotel['city']==city.lower()]
    citybase=citybase.sort_values(by='starrating',ascending=False)
    citybase.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    if(citybase.empty==0):
        hname=citybase[['hotelname','starrating','address','roomamenities','ratedescription']]
        x = hname[['hotelname','starrating','address','roomamenities','ratedescription']].head()
        data_list = x.to_dict(orient='records')
        return data_list
    else:
        print('No Hotels Available')
        return 'No Hotels Available'

room_no=[
     ('king',2),
   ('queen',2), 
    ('triple',3),
    ('master',3),
   ('family',4),
   ('murphy',2),
   ('quad',4),
   ('double-double',4),
   ('mini',2),
   ('studio',1),
    ('junior',2),
   ('apartment',4),
    ('double',2),
   ('twin',2),
   ('double-twin',4),
   ('single',1),
     ('diabled',1),
   ('accessible',1),
    ('suite',2),
    ('one',2)
   ]
def calc():
    guests_no=[]
    for i in range(hotel.shape[0]):
        temp=hotel['roomtype'][i].lower().split()
        flag=0
        for j in range(len(temp)):
            for k in range(len(room_no)):
                if temp[j]==room_no[k][0]:
                    guests_no.append(room_no[k][1])
                    flag=1
                    break
            if flag==1:
                break
        if flag==0:
            guests_no.append(2)
    hotel['guests_no']=guests_no

def requirementbased(city,number,features):
    calc()
    hotel['roomamenities']=hotel['roomamenities'].str.replace(': ;',',')

    hotel['city']=hotel['city'].str.lower()
    hotel['roomamenities']=hotel['roomamenities'].str.lower()
    features=features.lower()
    features_tokens=word_tokenize(features)  
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set=set()
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased=hotel[hotel['city']==city.lower()]
    reqbased=reqbased[reqbased['guests_no']==number]
    reqbased=reqbased.set_index(np.arange(reqbased.shape[0]))
    l1 =[];l2 =[];cos=[];
    #print(reqbased['roomamenities'])
    for i in range(reqbased.shape[0]):
        temp_tokens=word_tokenize(reqbased['roomamenities'][i])
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        #print(rvector)
        cos.append(len(rvector))
    reqbased['similarity']=cos
    reqbased=reqbased.sort_values(by='similarity',ascending=False)
    reqbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    x = reqbased[['city','hotelname','roomtype','guests_no','starrating','address','roomamenities','ratedescription','similarity']].head(10)
    data_list = x.to_dict(orient='records')
    return data_list

class cityBased(BaseModel):
    city : str

class requirementBased(BaseModel):
    city : str
    number: int
    features : str

@app.post('/city_based_recommendation')
def city_based_recommendation(data:cityBased):
    return citybased(data.city)

@app.post('/requirement_based_recommendation')
def requirement_based_recommendation(data:requirementBased):
    return requirementbased(data.city, data.number, data.features)