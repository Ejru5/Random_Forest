from fastapi import FastAPI

app = FastAPI()

@app.get("/")  #http://localhost:8000 GET
def index():
    return {"message": "Hello World"}

@app.get("/test")
def index():
    return {"message": "Test API"}

