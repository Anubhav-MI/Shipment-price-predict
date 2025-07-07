# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from shipment.exception.exception import ShipmentException
from shipment.pipeline.training_pipeline import run_training_pipeline
from shipment.logging.logger import logging
from shipment.components.model_predictor import shippingData, CostPredictor

import os

app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/form", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})



@app.get("/train")
async def train_route():
    try:
        result = run_training_pipeline()
        return result
    except Exception as e:
        return {"error": str(e)}




@app.post("/predict")
async def predict(
    customerId: str = Form(...),
    artistName: str = Form(...),
    artist: float = Form(...),
    height: float = Form(...),
    width: float = Form(...),
    weight: float = Form(...),
    material: str = Form(...),
    priceOfSculpture: float = Form(...),
    baseShippingPrice: float = Form(...),
    international: str = Form(...),
    expressShipment: str = Form(...),
    installationIncluded: str = Form(...),
    transport: str = Form(...),
    fragile: str = Form(...),
    customerInformation: str = Form(...),
    remoteLocation: str = Form(...),
    scheduledDate: str = Form(...),   # Not used in prediction
    deliveryDate: str = Form(...),    # Not used in prediction
    customerLocation: str = Form(...) # Not used in prediction
):
    try:
        # Step 1: Wrap input into shippingData class
        shipping_data = shippingData(
            artistReputation=artist,
            height=height,
            width=width,
            weight=weight,
            material=material,
            priceOfSculpture=priceOfSculpture,
            baseShippingPrice=baseShippingPrice,
            international=international,
            expressShipment=expressShipment,
            installationIncluded=installationIncluded,
            transport=transport,
            fragile=fragile,
            customerInformation=customerInformation,
            remoteLocation=remoteLocation
        )

        # print("data reecived")
        # Log the input data
        logging.info(f"Received input data: {shipping_data}")

        # Step 2: Convert to DataFrame
        input_df = shipping_data.get_input_data_frame()

        # Step 3: Load model and predict
        predictor = CostPredictor()
        prediction = predictor.predict(input_df)

        return {
            "customer_id": customerId,
            "artist_name": artistName,
            "estimated_shipping_cost": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}

#to start server
# python -m uvicorn backend:app --reload