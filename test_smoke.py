from generate_data import make_data

def test_generated_sensor_data_schema():
    data = make_data(n=120, seed=11)
    assert {"temperature_c", "vibration_mm_s", "pressure_kpa", "runtime_hours", "failure_next_24h"} <= set(data.columns)
    assert len(data) == 120
    assert set(data["failure_next_24h"].unique()) <= {0, 1}
