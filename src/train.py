import os
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "..", "mlflow.db")
sqlite_uri = f"sqlite:///{db_path}"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Iris_Classification")


def train_model():
    with open(os.path.join(BASE_DIR, "..", "params.yaml"), "r") as f:
        params = yaml.safe_load(f)["train"]

    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "iris.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        metrics_path = os.path.join(BASE_DIR, "..", "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f)

        joblib.dump(model, os.path.join(BASE_DIR, "..", "models", "model.pkl"))
        print(f"Модель обучена. Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
