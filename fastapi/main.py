from fastapi import FastAPI

app = FastAPI()
@app.get("/") # @ - decorator , "get('/hello')"--> http://localhost:8000/hello
def index(): # index is arbitary
    return {"Message" : "Hello World"}

@app.get("/test") # @ - decorator , "get('/test')"--> http://localhost:8000/test
def test():
    a = 5
    return {"Message" : "Testing World",
            "Value" : a}