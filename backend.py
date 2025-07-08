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

@app.get("/form", response_class=HTMLResponse)
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
    customerId: str = Form(...),
    artistName: str = Form(...),
    artist: float = Form(...),
    material: str = Form(...),
    priceOfSculpture: float = Form(...),
    height: float = Form(...),
    width: float = Form(...),
    weight: float = Form(...),
    baseShippingPrice: float = Form(...),
    international: str = Form(...),
    expressShipment: str = Form(...),
    installationIncluded: str = Form(...),
    transport: str = Form(...),
    fragile: str = Form(...),
    customerInformation: str = Form(...),
    remoteLocation: str = Form(...),
    customerLocation: str = Form(...),
    scheduledDate: str = Form(...),
    deliveryDate: str = Form(...)
):
    try:
        # Extract month and year from deliveryDate
        delivery_dt = datetime.strptime(deliveryDate, "%Y-%m-%d")
        month = delivery_dt.month
        year = delivery_dt.year

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Artist Reputation': artist,
            'Height': height,
            'Width': width,
            'Base Shipping Price': baseShippingPrice,
            'Month': month,
            'Year': year,
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
        estimated_cost = float(np.expm1(prediction[0]))  # if log1p was used during training

        # Return prediction result as JSON
        return JSONResponse({"estimated_shipping_cost": round(estimated_cost, 2)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Run the server (use this if running directly)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




#to start server
# python -m uvicorn backend:app --reload