from typing import Any, Dict, Optional
from mlservice.core import MLModel, ModelParams

class DummyModel(MLModel):
    def __init__(self, params: ModelParams):
        super().__init__(params)
        self.model = None

    def _predict(self, data: Any) -> Dict[str, Any]:
        return {"message": "Dummy model prediction"}
    
    def _train(self, train_data: Any, eval_data: Optional[Any] = None) -> None:
        """Implementation of model training logic."""
        pass

    def _evaluate(self, data):
        """Implementation of evaluation logic."""
        return {"accuracy": 1.0}  # Dummy metric

    def __str__(self):
        return f"DummyModel: {self.params}"
 