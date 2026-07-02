from sklearn.datasets import load_breast_cancer
import pandas as pd

from src.preprocess import split_data
from src.tune import tune_svm
from src.train import build_model, train_model
from src.evaluate import evaluate_model
from src.save_model import save_model


def main():

    # Load data
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Hyperparameter tuning
    tuning_results = tune_svm(
        X_train,
        y_train
    )

    best_params = tuning_results["best_params"]

    # Build & train
    model = build_model(**best_params)

    model = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluation
    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nEvaluation Results")
    print("-" * 30)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")

    # Save model
    save_model(
        model,
        model_name="svm_optuna.pkl"
    )

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
