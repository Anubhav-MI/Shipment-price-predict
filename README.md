# 🚚 Shipment Price Prediction

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Regression-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> **A sophisticated machine learning pipeline for predicting sculpture shipping costs** 🎨

This project predicts the shipping cost of sculptures based on various features such as artist reputation, dimensions, material, and shipping options. It uses a comprehensive machine learning pipeline for data ingestion, validation, transformation, model training, prediction, and evaluation.

## ✨ Features

| Component | Description | Status |
|-----------|-------------|---------|
| 📥 **Data Ingestion** | Loads and prepares shipment data for processing | ✅ Complete |
| 🔍 **Data Validation** | Validates the ingested data for quality and schema compliance | ✅ Complete |
| 🔄 **Data Transformation** | Transforms and preprocesses data for model training | ✅ Complete |
| 🤖 **Model Training** | Trains a regression model to predict shipment prices | ✅ Complete |
| 🎯 **Prediction** | Predicts shipment cost for sample inputs | ✅ Complete |
| 📊 **Model Evaluation** | Evaluates the trained model on test data | ✅ Complete |

## 📁 Project Structure

```
📦 Shipment-price-predict
├── 📄 app.py                        # 🚀 Main script to run the pipeline
├── 📄 backend.py                    # 🌐 Backend server implementation
├── 📄 form.html                     # 📝 Web form for user input
├── 📄 requirements.txt              # 📋 Python dependencies
├── 📄 setup.py                      # ⚙️ Package setup configuration
├── 📂 shipment/
│   ├── 📂 components/               # 🔧 Pipeline components
│   │   ├── 📄 data_ingestion.py     # 📥 Data loading & preparation
│   │   ├── 📄 data_validation.py    # 🔍 Data quality validation
│   │   ├── 📄 data_transformation.py # 🔄 Data preprocessing
│   │   ├── 📄 model_trainer.py      # 🤖 Model training logic
│   │   └── 📄 model_predictor.py    # 🎯 Prediction functionality
│   ├── 📂 entity/                   # 🏗️ Entity and config classes
│   ├── 📂 exception/                # ❌ Custom exception handling
│   ├── 📂 logging/                  # 📝 Logging utilities
│   ├── 📂 pipeline/
│   │   └── 📄 training_pipeline.py  # 🔄 Pipeline orchestration
│   └── 📂 utils/                    # 🛠️ Utility functions
├── 📂 shipment_data/
│   └── 📂 data/
│       └── 📄 shipment.csv          # 📊 Training dataset
├── 📂 data_schema/                  # 📋 Data schema definitions
└── 📂 Artifacts/                    # 📦 Generated artifacts & models
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **📥 Clone the repository:**
   ```bash
   git clone https://github.com/Anubhav-MI/Shipment-price-predict.git
   cd Shipment-price-predict
   ```

2. **📦 Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **🏃‍♂️ Run the pipeline:**
   ```bash
   python app.py
   ```

### Alternative: Web Interface
Run the backend server for web-based predictions:
```bash
python backend.py
```

## 📊 Expected Output

When you run the pipeline, you'll see:

```
🚚 Starting data ingestion...
🔍 Starting data validation...
🔄 Data transformation in progress...
🤖 Model training initiated...
📊 Model evaluation complete...
✅ Sample shipment cost prediction: $2,847.50
```

The pipeline will generate:
- 📁 **Artifacts/** - Contains trained models and processed data
- 📝 **Logs/** - Detailed execution logs
- 📊 **Reports/** - Data validation and drift reports

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

Docker Link:https://hub.docker.com/repository/registry-1.docker.io/anubhavmi/shipment_price_docker/general
