import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import os

class ModelEvaluation:
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model file not found at {self.model_path}")
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)

    def load_test_data(self):
        if not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"❌ Test data not found at {self.test_data_path}")
        return pd.read_csv(self.test_data_path)

    def evaluate(self):
        model = self.load_model()
        df = self.load_test_data()

        # Drop columns not used during training
        drop_cols = ['Customer Id', 'Scheduled Date', 'Delivery Date']
        df = df.drop(columns=drop_cols, errors='ignore')

        df = df.dropna()

        if 'Cost' not in df.columns:
            raise ValueError("❌ 'Cost' column not found in test data.")

        y_test = df['Cost']
        X_test = df.drop('Cost', axis=1)

        try:
            preds = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"❌ Model prediction failed: {e}")

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print("✅ Evaluation Complete")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        return {'rmse': rmse, 'r2_score': r2}