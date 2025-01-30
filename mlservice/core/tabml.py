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
from .ml import MLModel




class TabModel(MLModel):
    """Base class for regression models."""

    def __init__(self, params=None):
        super().__init__(params)

    def _set_feature_columns(self, columns: List[str]):
        """Set feature columns used by the model."""
        if 'columns' not in self.params:
            self.params['columns'] = {}
        self.params["columns"]["features"] = columns

    @property
    def feature_columns(self) -> List[str]:
        """Return feature columns used by the model."""
        return self.params.get("columns", {}).get("features", [])

    @property
    def target_column(self) -> str:
        """Return target column used by the model."""
        return self.params.get("columns", {}).get("target", "target")

    def _infer_features_columns(self, columns=None)->List[str]:
        features = self.feature_columns
        if features:
            return features
        if columns is None:
            return []
        return [col for col in columns if col not in [self.target_column, self.prediction_column, self.predict_proba_column]]

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
        df = self._predict(data)
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
        df = self._predict(data)
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
