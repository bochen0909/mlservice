# PyMLService

A FastAPI-based service for deploying machine learning models with dynamic route registration capabilities. PyMLService provides a flexible and extensible framework for serving ML models through REST APIs.

## Features

- **Dynamic Route Registration**: Register new endpoints dynamically using a decorator-based registry pattern
- **ML Model Support**: Built-in support for tabular machine learning models (regression and classification)
- **External Route Integration**: Easily extend functionality by importing external route modules
- **API Documentation**: Automatic Swagger/OpenAPI documentation at `/docs` endpoint
- **Scikit-learn Integration**: Built-in support for scikit-learn models with extensible base classes
- **CI/CD Ready**: Integrated GitHub Actions workflows for testing and deployment

## Installation

### Prerequisites

- Python >3.10
- Poetry (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mlservice
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Starting the Server

Run the server with default settings:
```bash
# Note: Even though the PyPI package is named 'pymlservice', 
# we still use the internal module name 'mlservice'
poetry run python -m mlservice.main
```

Custom configuration:
```bash
poetry run python -m mlservice.main --host 0.0.0.0 --port 8000 --external-routines external_routes.sklearn
```

### Adding ML Models

1. Create a new model class inheriting from `TabRegression` or `TabClassification`:

```python
from mlservice.core.tabml import TabRegression
from mlservice.core.ml import model_endpoints

@model_endpoints("sklearn/ridge")
class RidgeModel(TabRegression):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = Ridge(alpha=self.hyperparameters.get("alpha", 1.0))

    def _train(self, train_data, eval_data=None):
        # Implementation
        pass

    def _predict(self, data):
        # Implementation
        pass
```

2. Register the model routes using the `model_endpoints` decorator
3. Import the model module when starting the server

### API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
mlservice/
├── mlservice/
│   ├── core/            # Core functionality
│   │   ├── ml.py       # Base ML model classes
│   │   ├── registry.py # Route registration system
│   │   ├── router.py   # Core router setup
│   │   └── tabml.py    # Tabular ML model support
│   └── main.py         # FastAPI application setup
├── external_routes/     # External route modules
│   ├── sklearn/        # Scikit-learn model implementations
│   └── demo/           # Example implementations
├── tests/              # Test suite
├── poetry.lock         # Lock file for dependencies
└── pyproject.toml      # Project configuration
```

## Development

### Running Tests

Run the test suite using pytest:
```bash
poetry run pytest
```

With coverage report:
```bash
poetry run pytest --cov
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Write or update tests
5. Submit a pull request

## License

[Add License Information]
