"""
Tests for utility functions.
"""
import os
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open
from mlservice.core.utils import load_data, load_model

def test_load_data_none():
    """Test load_data with None path."""
    assert load_data(None) is None

def test_load_data_csv(tmp_path):
    """Test load_data with CSV file."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(csv_path, index=False)
    
    result = load_data(str(csv_path))
    pd.testing.assert_frame_equal(result, df)

def test_load_data_json(tmp_path):
    """Test load_data with JSON file."""
    json_path = tmp_path / "test.json"
    test_data = {"key": "value", "numbers": [1, 2, 3]}
    
    with open(json_path, "w") as f:
        json.dump(test_data, f)
    
    result = load_data(str(json_path))
    assert result == test_data

def test_load_data_other_format(tmp_path):
    """Test load_data with unsupported format."""
    test_path = tmp_path / "test.txt"
    test_content = "test content"
    
    # Create the test file
    with open(test_path, "w") as f:
        f.write(test_content)
    
    result = load_data(str(test_path))
    assert result == str(test_path)

def test_load_data_not_found():
    """Test load_data with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv")

def test_load_model_directory_not_found():
    """Test load_model with nonexistent directory."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model("nonexistent/model/path")
    assert "Model directory not found" in str(exc_info.value)

def test_load_model_file_not_found(tmp_path):
    """Test load_model with empty model directory."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model(str(tmp_path))
    assert "Model files missing in" in str(exc_info.value)

@patch('joblib.load')
def test_load_model_corrupt(mock_joblib_load, tmp_path):
    """Test load_model with corrupted model file."""
    # Create model directory and empty model file
    model_dir = tmp_path
    model_file = model_dir / "model.joblib"
    model_file.touch()
    
    # Mock joblib.load to raise an exception
    mock_joblib_load.side_effect = Exception("Corrupted model file")
    
    with pytest.raises(ValueError) as exc_info:
        load_model(str(model_dir))
    assert "Error loading model file" in str(exc_info.value)
