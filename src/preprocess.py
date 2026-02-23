from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def preprocess_data(df):
    """
    Split dataset and create preprocessing pipeline.
    """

    # Target column in California dataset
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling pipeline
    preprocessor = Pipeline([
        ("scaler", StandardScaler())
    ])

    return X_train, X_test, y_train, y_test, preprocessor