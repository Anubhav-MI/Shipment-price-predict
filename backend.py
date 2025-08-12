from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import uvicorn
from shipment.pipeline.training_pipeline import run_training_pipeline  # Assuming this is your training script

import os


# Load trained model pipeline
with open("saved_models/shipping_price_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Mount static files (images, CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")
# Define the root route

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/train")
async def train_route():
    try:
        result = run_training_pipeline()
        return result
    except Exception as e:
        return {"error": str(e)}




@app.post("/predict")
async def predict_shipping_cost(
    artist: float = Form(...),
    height: float = Form(...),
    width: float = Form(...),
    baseShippingPrice: float = Form(...),
    priceOfSculpture: float = Form(...),
    weight: float = Form(...),
    material: str = Form(...),
    international: str = Form(...),
    expressShipment: str = Form(...),
    installationIncluded: str = Form(...),
    transport: str = Form(...),
    fragile: str = Form(...),
    customerInformation: str = Form(...),
    remoteLocation: str = Form(...),
    shipmentMonth: str = Form(...),
    shipmentYear: int = Form(...)
):
    try:
        # Convert month string to number
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month = month_mapping.get(shipmentMonth)
        if not month:
            return JSONResponse({"error": "Invalid month name."}, status_code=400)

        # Construct input data
        input_data = pd.DataFrame([{
            'Artist Reputation': artist,
            'Height': height,
            'Width': width,
            'Base Shipping Price': baseShippingPrice,
            'Month': month,
            'Year': shipmentYear,
            'Material': material,
            'International': international,
            'Express Shipment': expressShipment,
            'Installation Included': installationIncluded,
            'Transport': transport,
            'Fragile': fragile,
            'Customer Information': customerInformation,
            'Remote Location': remoteLocation,
            'Price Of Sculpture': priceOfSculpture,
            'Weight': weight
        }])

        # Make prediction
        prediction = pipeline.predict(input_data)
        estimated_cost = float(np.expm1(prediction[0]))  # if log1p was used

        return JSONResponse({"estimated_shipping_cost": round(estimated_cost, 2)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
# Run the server (use this if running directly)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




#to start server
# python -m uvicorn backend:app --reload