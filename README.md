# Shipment Price Prediction

This project predicts the shipping cost of sculptures based on various features such as artist reputation, dimensions, material, and shipping options. It uses a machine learning pipeline for data ingestion, validation, transformation, model training, prediction, and evaluation.

## Features
- **Data Ingestion:** Loads and prepares shipment data for processing.
- **Data Validation:** Validates the ingested data for quality and schema compliance.
- **Data Transformation:** Transforms and preprocesses data for model training.
- **Model Training:** Trains a regression model to predict shipment prices.
- **Prediction:** Predicts shipment cost for a sample input.
- **Model Evaluation:** Evaluates the trained model on test data.

## Project Structure
```
├── app.py                        # Main script to run the pipeline
├── shipment/
│   ├── components/               # Pipeline components (ingestion, validation, etc.)
│   ├── entity/                   # Entity and config classes
│   ├── exception/                # Custom exception handling
│   ├── logging/                  # Logging utilities
│   ├── pipeline/
│   │   └── training_pipeline.py  # Pipeline orchestration
│   └── utils/                    # Utility functions
├── requirements.txt              # Python dependencies
```

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the pipeline:**
   ```bash
   python app.py
   ```

## Example Output
- Data ingestion, validation, transformation, and model training logs will be printed to the console.
- A sample shipment cost prediction will be displayed if the model is trained successfully.
- Model evaluation metrics will be shown at the end.

## Sample Prediction Input
```
artistReputation=5
height=12
width=10
weight=8
material="Marble"
priceOfSculpture=20000
baseShippingPrice=500
international="Yes"
expressShipment="No"
installationIncluded="Yes"
transport="Airways"
fragile="Yes"
customerInformation="Wealthy"
remoteLocation="No"
```

## License
This project is licensed under the MIT License.
