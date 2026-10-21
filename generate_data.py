from pathlib import Path
import numpy as np
import pandas as pd

def make_data(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    temperature = rng.normal(70, 10, n).clip(35, 115)
    vibration = rng.gamma(2.2, 1.2, n).clip(0.1, 12)
    pressure = rng.normal(100, 12, n).clip(55, 145)
    runtime_hours = rng.integers(100, 12000, n)
    wear = (runtime_hours / 12000) * 0.8 + (vibration / 12) * 0.7 + (temperature > 90) * 0.8
    probability = 1 / (1 + np.exp(-(wear - 1.25) * 2.1))
    failure = rng.binomial(1, probability)
    return pd.DataFrame({
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(2),
        "pressure_kpa": pressure.round(2),
        "runtime_hours": runtime_hours,
        "failure_next_24h": failure
    })

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "sensor_readings.csv"
    out.parent.mkdir(exist_ok=True)
    make_data().to_csv(out, index=False)
    print(f"Wrote {out}")
