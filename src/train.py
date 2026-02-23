import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from src.config import MODELS_DIR


def train_models(X_train, y_train, preprocessor):
    """
    Train multiple regression models and save them.
    """

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1)
    }

    trained_models = {}

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(pipeline, model_path)

        print(f"{name.upper()} model saved at {model_path}")

        trained_models[name] = pipeline

    return trained_models