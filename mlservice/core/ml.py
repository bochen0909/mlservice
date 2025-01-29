import os
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd
import joblib
from pydantic import BaseModel
from .utils import load_data, load_model

class ModelParams(BaseModel):
    pass



class MLModel(ABC):
    """Base class for ML models with training, prediction, and evaluation capabilities."""
    
    def __init__(self, params: ModelParams) -> None:
        self.params = params
        
    def _get_model_dir(self, name: str, version: str) -> Path:
        """Generate model directory path with versioning."""
        ml_home = os.getenv('ML_HOME')
        if not ml_home:
            raise ValueError("ML_HOME environment variable not set")
            
        today = datetime.now()
        model_dir = Path(ml_home) / "models" / name / version / \
                   str(today.year) / f"{today.month:02d}" / f"{today.day:02d}" / str(uuid.uuid4())
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
        
    def train(self, train_path: str, eval_path: str = None, test_path: str = None) -> Dict[str, Any]:
        """Train the model and save artifacts.
        
        Args:
            train_path: Path to training data
            eval_path: Optional path to evaluation data
            test_path: Optional path to test data
            
        Returns:
            Dict containing training metrics and metadata
        """
        # Load data
        train_data = load_data(train_path)
        eval_data = load_data(eval_path)
        test_data = load_data(test_path)
        
        # Train model
        self._train(train_data, eval_data)
        
        # Evaluate on available datasets
        metrics = {}
        if train_data is not None:
            metrics['train'] = self._evaluate(train_data)
        if eval_data is not None:
            metrics['validation'] = self._evaluate(eval_data)
        if test_data is not None:
            metrics['test'] = self._evaluate(test_data)
        
        # Save model and metadata
        model_dir = self._get_model_dir('model_name', 'model_version')
        
        # Save model
        joblib.dump(self, model_dir / "model.joblib")
        
        # Save parameters
        with open(model_dir / "params.json", 'w') as f:
            json.dump(self.params.dict(), f, indent=2)
            
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'train_path': train_path,
            'eval_path': eval_path,
            'test_path': test_path,
            'metrics': metrics
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata

    @abstractmethod
    def _train(self, train_data: Any, eval_data: Optional[Any] = None) -> None:
        """Implementation of model training logic."""
        pass
        
    def predict(self, data_path: str) -> Dict[str, Any]:
        """Make predictions on new data.
        
        Args:
            data_path: Path to input data
            
        Returns:
            Dict containing predictions
        """
        data = load_data(data_path)
        return self._predict(data)
        
    @abstractmethod
    def _predict(self, data: Any) -> Dict[str, Any]:
        """Implementation of prediction logic."""
        pass
        
    def evaluate(self, data_path: str) -> Dict[str, float]:
        """Evaluate model on new data.
        
        Args:
            data_path: Path to evaluation data
            
        Returns:
            Dict containing evaluation metrics
        """
        data = load_data(data_path)
        return self._evaluate(data)
        
    @abstractmethod
    def _evaluate(self, data: Any) -> Dict[str, Any]:
        """Implementation of evaluation logic."""
        pass
