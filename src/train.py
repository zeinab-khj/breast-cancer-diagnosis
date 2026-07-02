from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import mlflow

mlflow.set_experiment("breast-cancer-svm")


def build_model(C=1.0, kernel="rbf", gamma="scale"):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, kernel=kernel, gamma=gamma))
    ])
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)
