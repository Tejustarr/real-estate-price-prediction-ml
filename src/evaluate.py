import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using R2 and RMSE.
    """

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return r2, rmse