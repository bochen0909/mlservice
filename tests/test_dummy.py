import pytest
import os
from pathlib import Path
import pandas as pd
import joblib
from external_routes.mldemo.dummy import DummyModel

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def sample_data():
    return pd.DataFrame({'col1': [1, 2, 3]})

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

    print(dir(loaded_model))
    
    # Check if the loaded model is fitted
    assert loaded_model.fitted_, f"Loaded model should be marked as fitted, but got {loaded_model.fitted_}"
    
    # Test loaded model predictions
    predict_path = tmp_path / "predict.csv"
    sample_data.to_csv(predict_path)
    prediction = loaded_model.predict(str(predict_path))
    
    assert prediction == {"message": "Dummy model prediction"}
    assert isinstance(loaded_model, DummyModel)
