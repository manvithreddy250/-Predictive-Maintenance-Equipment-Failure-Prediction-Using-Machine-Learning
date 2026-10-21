# Predictive Maintenance & Failure Detection

End-to-end binary classification demo using synthetic equipment sensor readings. Includes data generation, train/test split, Random Forest model, classification metrics, saved model, and Streamlit inference dashboard.

## Run
```bash
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

## Inputs
- temperature_c
- vibration_mm_s
- pressure_kpa
- runtime_hours

## Outputs
- Estimated probability of failure in the next 24 hours
- Accuracy, precision, recall, F1 and ROC-AUC on a held-out test split
- Basic sensor plots

## Important
The dataset is synthetic and intentionally simple. Metrics demonstrate the workflow, not real-world predictive performance. Do not use for operational or safety-critical maintenance decisions.

## Smoke test

Run `python -m pytest test_smoke.py` after installing the requirements.
