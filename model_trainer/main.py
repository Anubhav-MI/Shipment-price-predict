import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import joblib

# === Setup ===
TRAIN_FILE_PATH = r"C:\Users\Welcome\Downloads\train.csv"
TEST_FILE_PATH = r"C:\Users\Welcome\Downloads\test.csv"
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === Load Data ===
train_df = pd.read_csv(TRAIN_FILE_PATH).dropna()
test_df = pd.read_csv(TEST_FILE_PATH).dropna()

# === Encode ===
def encode_dataframe(train_df, test_df):
    label_encoders = {}
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            le = LabelEncoder()
            all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
            le.fit(all_values)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
    return train_df, test_df

train_df, test_df = encode_dataframe(train_df, test_df)

# === Features & Target ===
X_train = train_df.drop(columns=["Cost"])
y_train = train_df["Cost"]
X_test = test_df.drop(columns=["Cost"])
y_test = test_df["Cost"]

# === Base Models ===
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "KNeighbors": KNeighborsRegressor(),
    "SVR": SVR(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0)
}

# === Evaluate All Models ===
performance = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        performance.append({
            "model": name,
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "object": model
        })

        print(f"{name}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    except Exception as e:
        print(f"❌ {name} failed: {e}")

# === Determine Best Metric Based on Ranking Consistency ===
df = pd.DataFrame(performance)

# Rank each metric (lower is better for mse/mae, higher is better for r2)
df['r2_rank'] = df['r2'].rank(ascending=False)
df['mse_rank'] = df['mse'].rank()
df['mae_rank'] = df['mae'].rank()

# Compute average rank for each metric
avg_ranks = {
    "r2": df['r2_rank'].mean(),
    "mse": df['mse_rank'].mean(),
    "mae": df['mae_rank'].mean()
}

# Select the metric with the lowest average rank
best_metric = min(avg_ranks, key=avg_ranks.get)
print(f"\n✅ Best metric selected based on ranking consistency: {best_metric.upper()}")

# === Sort Models Based on Best Metric ===
sorted_models = sorted(performance, key=lambda x: x[best_metric], reverse=(best_metric == "r2"))

print("\n📊 Sorted Models Based on Best Metric:")
for i, model_info in enumerate(sorted_models, 1):
    print(f"{i}. {model_info['model']} → {best_metric.upper()} = {model_info[best_metric]:.4f}")

# === Pick Top 3 Models ===
top_models = sorted_models[:3]
print("\n🔝 Top 3 Models:")
for m in top_models:
    print(f"{m['model']} → {best_metric.upper()} = {m[best_metric]:.4f}")

# === Hyperparameter Grids ===
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    },
    "CatBoost": {
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "iterations": [100, 200]
    },
    "DecisionTree": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "AdaBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1.0]
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7]
    },
    "SVR": {
        "kernel": ['linear', 'rbf'],
        "C": [1, 10]
    },
    "LinearRegression": {}
}

# === Select Scorer for GridSearchCV Based on Best Metric ===
scorer_map = {
    "r2": "r2",
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False)
}
scorer = scorer_map[best_metric]

# === Tune Top 3 Models ===
best_score = -np.inf
best_model = None
best_model_name = ""

for entry in top_models:
    name = entry['model']
    base_model = entry['object']

    if name in param_grids:
        print(f"\n🔍 Tuning {name} using {best_metric.upper()}...")
        try:
            grid = GridSearchCV(base_model, param_grids[name], scoring=scorer, cv=3)
            grid.fit(X_train, y_train)
            tuned_model = grid.best_estimator_

            y_pred = tuned_model.predict(X_test)
            final_score = {
                "r2": r2_score(y_test, y_pred),
                "mse": -mean_squared_error(y_test, y_pred),
                "mae": -mean_absolute_error(y_test, y_pred)
            }[best_metric]

            print(f"✅ {name} tuned {best_metric.upper()}: {abs(final_score):.4f}")

            if final_score > best_score:
                best_score = final_score
                best_model = tuned_model
                best_model_name = name

        except Exception as e:
            print(f"❌ Tuning failed for {name}: {e}")

# === Save Best Tuned Model ===
if best_model:
    model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_tuned_{best_model_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n✅ Final Best Model: {best_model_name} (Tuned {best_metric.upper()} = {abs(best_score):.4f})")
    print(f"📦 Saved to: {model_path}")
else:
    print("❌ No model successfully tuned.")
