from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from generate_data import make_data

BASE = Path(__file__).parent
DATA = BASE / "data" / "sensor_readings.csv"
MODEL = BASE / "model.joblib"

def main():
    if DATA.exists():
        df = pd.read_csv(DATA)
    else:
        df = make_data()
        DATA.parent.mkdir(exist_ok=True)
        df.to_csv(DATA, index=False)
    features = ["temperature_c", "vibration_mm_s", "pressure_kpa", "runtime_hours"]
    X, y = df[features], df["failure_next_24h"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "classification_report": classification_report(y_test, pred, zero_division=0)
    }
    joblib.dump({"model": model, "features": features}, MODEL)
    (BASE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved model to {MODEL}")

if __name__ == "__main__":
    main()
