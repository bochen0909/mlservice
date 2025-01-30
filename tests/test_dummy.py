import pytest
import os
from pathlib import Path
import pandas as pd
import joblib
from fastapi.testclient import TestClient
from external_routes.mldemo.dummy import DummyModel
from mlservice.main import setup_routes, app
# Existing fixtures
@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def sample_data():
    return pd.DataFrame({'col1': [1, 2, 3]})

@pytest.fixture
def client():
    setup_routes(['external_routes'])
    return TestClient(app)

# Existing model tests
def test_train(dummy_model, sample_data, tmp_path):
    # Set ML_HOME for testing
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Create temporary data files
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    
    # Test training
    metadata = dummy_model.train(str(train_path))
    
    # Verify model is marked as fitted
    assert dummy_model.fitted_
    
    # Verify metadata was created
    assert 'timestamp' in metadata
    assert 'metrics' in metadata
    assert metadata['train_path'] == str(train_path)

def test_predict(dummy_model, sample_data, tmp_path):
    # Train model first
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    dummy_model.train(str(train_path))
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    
    prediction = dummy_model.predict(str(predict_path))
    assert prediction == {"message": "Dummy model prediction"}

def test_evaluate(dummy_model, sample_data, tmp_path):
    # Train model first
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    dummy_model.train(str(train_path))
    
    # Test evaluation
    eval_path = tmp_path / "eval.csv"
    sample_data.to_csv(eval_path)
    
    evaluation = dummy_model.evaluate(str(eval_path))
    assert isinstance(evaluation, dict)
    assert 'accuracy' in evaluation
    assert evaluation['accuracy'] == 1.0

def test_load_model(dummy_model, sample_data, tmp_path):
    # Train and save model
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    metadata = dummy_model.train(str(train_path))
    assert dummy_model.fitted_
    
    # Get model directory from metadata
    model_path = metadata['model_path']
    
    # Load model from saved file
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    # Check if the loaded model is fitted
    assert loaded_model.fitted_, f"Loaded model should be marked as fitted, but got {loaded_model.fitted_}"
    
    # Test loaded model predictions
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    prediction = loaded_model.predict(str(predict_path))
    
    assert prediction == {"message": "Dummy model prediction"}
    assert isinstance(loaded_model, DummyModel)

# API endpoint tests
def test_train_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Create test data
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    
    response = client.post(
        "/model/dummy/train",
        json={
            "train_path": str(train_path),
            "eval_path": None,
            "test_path": None,
            "params": None
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'timestamp' in data
    assert 'metrics' in data
    assert data['train_path'] == str(train_path)

def test_predict_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train model first
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    train_response = client.post(
        "/model/dummy/train",
        json={
            "train_path": str(train_path),
            "eval_path": None,
            "test_path": None,
            "params": None
        }
    )
    model_path = train_response.json()['model_path']
    
    # Test prediction endpoint
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    
    response = client.post(
        "/model/dummy/predict",
        json={
            "data_path": str(predict_path),
            "model_path": model_path
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"message": "Dummy model prediction"}

def test_eval_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train model first
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    train_response = client.post(
        "/model/dummy/train",
        json={
            "train_path": str(train_path),
            "eval_path": None,
            "test_path": None,
            "params": None
        }
    )
    model_path = train_response.json()['model_path']
    
    # Test evaluation endpoint
    eval_path = tmp_path / "eval.csv"
    sample_data.to_csv(eval_path)
    
    response = client.post(
        "/model/dummy/eval",
        json={
            "data_path": str(eval_path),
            "model_path": model_path
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert 'accuracy' in data
    assert data['accuracy'] == 1.0

def atest_error_handling(client, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Test with non-existent file
    response = client.post(
        "/model/dummy/predict",
        json={
            "data_path": "nonexistent.csv",
            "model_path": "nonexistent_model"
        }
    )
    assert response.status_code == 500
