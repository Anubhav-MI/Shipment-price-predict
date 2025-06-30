import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

class ModelEvaluation:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)

    def load_test_data(self):
        return pd.read_csv(self.test_data_path)

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=['object', 'bool']).columns:
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception as e:
                print(f"⚠️ Could not encode column {col}: {e}")
        return df

    def evaluate(self):
        model = self.load_model()
        df = self.load_test_data()

        # Drop irrelevant columns (adjust as needed for your project)
        drop_cols = ['Customer Id', 'Customer Information', 'Customer Location',
                     'Scheduled Date', 'Delivery Date', 'Artist Name']
        df = df.drop(columns=drop_cols, errors='ignore')

        df = df.dropna()  # Drop rows with missing values

        if 'Cost' not in df.columns:
            raise ValueError("❌ 'Cost' column not found in test data.")

        y_test = df['Cost']
        X_test = df.drop('Cost', axis=1)

        # Encode categorical columns
        X_test = self.encode_categoricals(X_test)

        # TEMP: Limit features to dummy model (expects only 3)
        USE_DUMMY_MODEL = True  # Set to False when using the real model

        if USE_DUMMY_MODEL:
            try:
                X_test = X_test[['Height', 'Width', 'Weight']]
            except KeyError:
                raise ValueError("❌ Dummy model expects 'Height', 'Width', and 'Weight' in test data.")

        # Predict and evaluate
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        print("✅ Evaluation Complete")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        return {'rmse': rmse, 'r2_score': r2}
