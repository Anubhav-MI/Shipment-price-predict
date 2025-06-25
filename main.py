# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from predict_model import predict_from_input

import os

app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/form", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
async def predict(
    Customer_Id: str = Form(...),
    Artist_Name: str = Form(...),
    Artist_Reputation: float = Form(...),
    Height: float = Form(...),
    Width: float = Form(...),
    Weight: float = Form(...),
    Material: str = Form(...),
    Price_Of_Sculpture: float = Form(...),
    Base_Shipping_Price: float = Form(...),
    International: str = Form(...),
    Express_Shipment: str = Form(...),
    Installation_Included: str = Form(...),
    Transport: str = Form(...),
    Fragile: str = Form(...),
    Customer_Information: str = Form(...),
    Remote_Location: str = Form(...),
    Scheduled_Date: str = Form(...),
    Delivery_Date: str = Form(...),
    Customer_Location: str = Form(...)
):
    input_data = {
        "Customer Id": Customer_Id,
        "Artist Name": Artist_Name,
        "Artist Reputation": Artist_Reputation,
        "Height": Height,
        "Width": Width,
        "Weight": Weight,
        "Material": Material,
        "Price Of Sculpture": Price_Of_Sculpture,
        "Base Shipping Price": Base_Shipping_Price,
        "International": International,
        "Express Shipment": Express_Shipment,
        "Installation Included": Installation_Included,
        "Transport": Transport,
        "Fragile": Fragile,
        "Customer Information": Customer_Information,
        "Remote Location": Remote_Location,
        "Scheduled Date": Scheduled_Date,
        "Delivery Date": Delivery_Date,
        "Customer Location": Customer_Location
    }

    # Replace with actual prediction call
    # predicted_cost = predict_from_input(input_data)
    predicted_cost = 1234.56  # Mock value

    return {"estimated_cost": predicted_cost}
