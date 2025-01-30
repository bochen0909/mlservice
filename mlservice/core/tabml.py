import os
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .utils import load_data


class MLModel(ABC):
    """Base class for ML models with training, prediction, and evaluation capabilities."""

    def __init__(self, params: Optional[Union[str | dict]] = None):
        if params is None:
            params = {}
        elif isinstance(params, str):
            params = json.loads(params)
        self.params = params
        self.fitted_ = False

    def _get_model_dir(self, name: str, version: str) -> Path:
        """Generate model directory path with versioning."""
        ml_home = os.getenv("ML_HOME")
        if not ml_home:
            raise ValueError("ML_HOME environment variable not set")

        today = datetime.now()
        model_dir = (
            Path(ml_home)
            / "models"
            / name
            / version
            / str(today.year)
            / f"{today.month:02d}"
            / f"{today.day:02d}"
            / str(uuid.uuid4())
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def train(
        self, train_path: str, eval_path: str = None, test_path: str = None
    ) -> Dict[str, Any]:
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
            metrics["train"] = self._evaluate(train_data)
        if eval_data is not None:
            metrics["validation"] = self._evaluate(eval_data)
        if test_data is not None:
            metrics["test"] = self._evaluate(test_data)

        # Save model and metadata
        model_dir = self._get_model_dir("model_name", "model_version")

        # Save model
        joblib.dump(self, model_dir / "model.joblib")

        # Save parameters
        with open(model_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "train_path": train_path,
            "eval_path": eval_path,
            "test_path": test_path,
            "model_path": str(model_dir),
            "metrics": metrics,
        }

        with open(model_dir / "metadata.json", "w") as f:
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

    def evaluate(self, data_path: str) -> Dict[str, Any]:
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


class TabModel(MLModel):
    """Base class for regression models."""

    def __init__(self, params=None):
        super().__init__(params)

    @property
    def feature_columns(self) -> List[str]:
        """Return feature columns used by the model."""
        return self.params.get("columns", {}).get("features", [])

    @property
    def target_column(self) -> str:
        """Return target column used by the model."""
        return self.params.get("columns", {}).get("target", "target")

    @property
    def prediction_column(self) -> str:
        """Return prediction column used by the model."""
        return self.params.get("columns", {}).get("prediction", "prediction")

    @property
    def predict_proba_column(self) -> str:
        """Return prediction probability column used by the model."""
        return self.params.get("columns", {}).get("predict_proba", "predict_proba")

    @property
    def categorical_columns(self) -> List[str]:
        """Return categorical columns used by the model."""
        return self.params.get("columns", {}).get("categorical", [])

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters used by the model."""
        return self.params.get("hyperparameters", {})


class TabRegression(TabModel):
    def __init__(self, params=None):
        super().__init__(params)

    def _evaluate(self, data):
        """Implementation of evaluation logic."""
        df = self.predict(data)
        gt = data[self.target_column].values
        pred = df[self.prediction_column].values

        mse = mean_squared_error(gt, pred)
        mae = mean_absolute_error(gt, pred)
        r2 = r2_score(gt, pred)
        return {"mse": mse, "mae": mae, "r2": r2}


class TabClassification(TabModel):
    def __init__(self, params=None):
        super().__init__(params)

    def _evaluate(self, data):
        """Implementation of evaluation logic."""
        df = self.predict(data)
        gt = data[self.target_column].values
        accuracy = None
        f1 = None
        precision = None
        recall = None
        auc_score = None
        if self.prediction_column in df.columns:
            pred = df[self.prediction_column].values
            accuracy = accuracy_score(gt, pred)
        if self.predict_proba_column in df.columns:
            pred_proba = df[self.predict_proba_column].values
            pred = (pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred)
            precision = precision_score(gt, pred)
            recall = recall_score(gt, pred)
            auc_score = roc_auc_score(gt, pred_proba)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc_score": auc_score,
        }
