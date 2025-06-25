# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from train_model import train_and_save_model
# from predict_model import predict_from_input

import os

app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/form", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# @app.get("/train")
# async def train():
#     msg = train_and_save_model()
#     return {"message": msg}

# @app.post("/predict")
# async def predict(
#     Pclass: int = Form(...),
#     Sex: str = Form(...),
#     Age: float = Form(...),
#     SibSp: int = Form(...),
#     Parch: int = Form(...),
#     Fare: float = Form(...),
#     Embarked: str = Form(...)
# ):
#     input_data = {
#         "Pclass": Pclass,
#         "Sex": Sex,
#         "Age": Age,
#         "SibSp": SibSp,
#         "Parch": Parch,
#         "Fare": Fare,
#         "Embarked": Embarked
#     }
#     pred = predict_from_input(input_data)
#     return {"prediction": "Survived" if pred == 1 else "Did not Survive"}
