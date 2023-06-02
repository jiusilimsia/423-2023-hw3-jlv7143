import pytest
import pandas as pd
import tempfile
from pathlib import Path
import os
import app  



def test_load_model_versions_happy_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        model_dir = dir_path / "model_v1"
        model_dir.mkdir()

        # Creating mock files
        with open(model_dir / "cloud_data.csv", 'w') as f:
            pass
        with open(model_dir / "cloud_classifier.pkl", 'wb') as f:
            pass

        result = app.load_model_versions(dir_path)
        assert len(result) == 1
        assert 'model_v1' in result

def test_load_model_versions_not_happy_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        result = app.load_model_versions(dir_path)
        assert len(result) == 0

def test_slider_values_happy_path():
    series = pd.Series([1, 2, 3, 4, 5])
    min_val, max_val, mean_val = app.slider_values(series)
    assert min_val == 1.0
    assert max_val == 5.0
    assert mean_val == 3.0

def test_slider_values_not_happy_path():
    series = pd.Series([])
    with pytest.raises(ValueError) as excinfo:
        app.slider_values(series)
    assert str(excinfo.value) == "Input series is empty."
