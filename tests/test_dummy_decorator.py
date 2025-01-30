import pytest
import os
import pandas as pd
import joblib
import pickle
from fastapi.testclient import TestClient
from external_routes.mldemo.dummy_decorator import DummyModel, model_endpoints
from mlservice.main import setup_routes, app

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

# Test decorator functionality
def test_model_endpoints_decorator():
    # Test that the decorator properly adds the router
    assert hasattr(DummyModel, 'router'), "Decorator should add router attribute to model class"
    assert DummyModel.router.prefix == "/model/dummy_decorator", "Router should have correct prefix"

# Model functionality tests
def test_train(dummy_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    
    metadata = dummy_model.train(str(train_path))
    
    assert dummy_model.fitted_
    assert 'timestamp' in metadata
    assert 'metrics' in metadata
    assert metadata['train_path'] == str(train_path)

def test_predict(dummy_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    dummy_model.train(str(train_path))
    
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    
    prediction_file_path = dummy_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    with open(prediction_file_path, 'rb') as f:
        prediction = pickle.load(f)
    assert prediction == {"message": "Dummy model prediction"}

def test_evaluate(dummy_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    dummy_model.train(str(train_path))
    
    eval_path = tmp_path / "eval.csv"
    sample_data.to_csv(eval_path)
    
    evaluation = dummy_model.evaluate(str(eval_path))
    assert isinstance(evaluation, dict)
    assert 'accuracy' in evaluation
    assert evaluation['accuracy'] == 1.0

def test_load_model(dummy_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    metadata = dummy_model.train(str(train_path))
    assert dummy_model.fitted_
    
    model_path = metadata['model_path']
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    assert loaded_model.fitted_
    assert hasattr(loaded_model.__class__, 'router'), "Loaded model class should retain router"
    
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    prediction_file_path = loaded_model.predict(str(predict_path))
    
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    with open(prediction_file_path, 'rb') as f:
        prediction = pickle.load(f)
    assert prediction == {"message": "Dummy model prediction"}
    assert isinstance(loaded_model, DummyModel)

# API endpoint tests
def test_train_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    
    response = client.post(
        "/model/dummy_decorator/train",
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
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    train_response = client.post(
        "/model/dummy_decorator/train",
        json={
            "train_path": str(train_path),
            "eval_path": None,
            "test_path": None,
            "params": None
        }
    )
    model_path = train_response.json()['model_path']
    
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    
    response = client.post(
        "/model/dummy_decorator/predict",
        json={
            "data_path": str(predict_path),
            "model_path": model_path
        }
    )
    
    assert response.status_code == 200
    prediction_file_path = response.json()
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    with open(prediction_file_path, 'rb') as f:
        prediction = pickle.load(f)
    assert prediction == {"message": "Dummy model prediction"}

def test_eval_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path)
    train_response = client.post(
        "/model/dummy_decorator/train",
        json={
            "train_path": str(train_path),
            "eval_path": None,
            "test_path": None,
            "params": None
        }
    )
    model_path = train_response.json()['model_path']
    
    eval_path = tmp_path / "eval.csv"
    sample_data.to_csv(eval_path)
    
    response = client.post(
        "/model/dummy_decorator/eval",
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

def test_error_handling(client, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    response = client.post(
        "/model/dummy_decorator/predict",
        json={
            "data_path": "nonexistent.csv",
            "model_path": "nonexistent_model"
        }
    )
    assert response.status_code == 500

# Additional decorator-specific tests
def test_multiple_model_endpoints():
    @model_endpoints("test_model1")
    class TestModel1(DummyModel):
        pass

    @model_endpoints("test_model2")
    class TestModel2(DummyModel):
        pass

    assert TestModel1.router.prefix == "/model/test_model1"
    assert TestModel2.router.prefix == "/model/test_model2"
    assert TestModel1.router != TestModel2.router
