import os
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, Type

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .utils import load_data, load_model



class TrainRequest(BaseModel):
    train_path: str
    eval_path: Optional[str] = None
    test_path: Optional[str] = None
    params: Optional[str] = None

class PredictRequest(BaseModel):
    data_path: str
    model_path: str

class EvalRequest(BaseModel):
    data_path: str
    model_path: str

def model_endpoints(model_name: str):
    """
    Decorator to create FastAPI endpoints for an MLModel class.
    
    Args:
        model_name: Name of the model for URL paths
        
    Returns:
        Decorator function that wraps MLModel class and adds router
    """
    def decorator(model_class):
        model_class.router = create_model_endpoints(model_class, model_name)
        return model_class
    return decorator

def create_model_endpoints(model_class: Type["MLModel"], model_name: str) -> APIRouter:
    """
    Create FastAPI endpoints for an MLModel class and register them with registry.
    
    Args:
        model_class: MLModel class to create endpoints for
        model_name: Name of the model for URL paths
        
    Returns:
        FastAPI router with /train, /predict, and /eval endpoints
    """
    from mlservice.core.registry import registry
    
    router = APIRouter(prefix=f"/model/{model_name}")
    
    @router.post("/train", tags=["ML Model"])
    async def train_model(request: TrainRequest):
        try:
            model = model_class(request.params)
            result = model.train(
                train_path=request.train_path,
                eval_path=request.eval_path,
                test_path=request.test_path
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/predict", tags=["ML Model"])
    async def predict(request: PredictRequest):
        try:
            # Load latest model
            model = load_model(request.model_path)
            if not isinstance(model, MLModel):
                raise ValueError("Loaded object is not an MLModel instance")
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            result = model.predict(data_path=request.data_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/eval", tags=["ML Model"])
    async def evaluate(request: EvalRequest):
        try:
            # Load latest model
            model = load_model(request.model_path)
            if not isinstance(model, MLModel):
                raise ValueError("Loaded object is not an MLModel instance")
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            result = model.evaluate(data_path=request.data_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Register routes in registry system
    for route in router.routes:
        methods = list(route.methods)  # Convert set to list
        registry._routes.append({
            'path': route.path_format,
            'methods': methods,
            'handler': route.endpoint,
            'kwargs': {
                'response_model': route.response_model,
                'status_code': route.status_code,
                'tags': route.tags
            }
        })
    
    return router



class MLModel(ABC):
    """Base class for ML models with training, prediction, and evaluation capabilities."""
    
    def __init__(self, params: Optional[Union[str|dict]] = None):
        if params is None:
            params = {}
        elif isinstance(params, str):
            params = json.loads(params)
        self.params = params
        self.fitted_ = False
    
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
        self.fitted_ = True

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
            json.dump(self.params, f, indent=2)
            
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'train_path': train_path,
            'eval_path': eval_path,
            'test_path': test_path,
            "model_path": str(model_dir),
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
        if not self.fitted_:
            raise ValueError("Model must be trained before prediction")
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
        if not self.fitted_:
            raise ValueError("Model must be trained before evaluation")
        data = load_data(data_path)
        return self._evaluate(data)
        
    @abstractmethod
    def _evaluate(self, data: Any) -> Dict[str, Any]:
        """Implementation of evaluation logic."""
        pass
