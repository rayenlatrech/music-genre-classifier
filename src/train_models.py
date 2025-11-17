import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


DATA_PATH = os.path.join("data", "gtzan_features.csv")
MODELS_DIR = "models"


def load_data():
    """
    Load the precomputed feature dataset and split into X (features) and y (labels).
    """
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Drop non-numeric / identifier columns
    X = df.drop(columns=["genre", "filename"])
    y = df["genre"]

    print("Feature matrix shape:", X.shape)
    print("Number of classes:", y.nunique())
    print("Classes:", sorted(y.unique()))

    return X, y


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Create a stratified train/test split to preserve genre proportions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keeps genre distribution similar in train and test
    )

    print("\nTrain size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def get_model_configs():
    """
    Define pipelines and hyperparameter grids for:
      - KNN
      - SVM (RBF kernel)
      - Random Forest
    """
    configs = []

    # ---- KNN ---- #
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])

    param_grid_knn = {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"]
    }

    configs.append(("KNN", pipe_knn, param_grid_knn))

    # ---- SVM (RBF) ---- #
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    param_grid_svm = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": [0.01, 0.1, 1.0]
    }

    configs.append(("SVM_RBF", pipe_svm, param_grid_svm))

    # ---- Random Forest ---- #
    pipe_rf = Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid_rf = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }

    configs.append(("RandomForest", pipe_rf, param_grid_rf))

    return configs


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Run GridSearchCV for each model, evaluate on test set,
    and return a summary with the best model.
    """
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_overall_model = None
    best_overall_name = None
    best_overall_f1 = -np.inf

    model_configs = get_model_configs()

    for name, pipeline, param_grid in model_configs:
        print(f"\n{'='*40}")
        print(f"Training model: {name}")
        print(f"{'='*40}")

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        print(f"\nBest params for {name}: {grid.best_params_}")
        print(f"Best CV f1_macro for {name}: {grid.best_score_:.4f}")

        best_model = grid.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nTest Accuracy for {name}: {acc:.4f}")
        print(f"Test Macro F1 for {name}: {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
        disp.plot(xticks_rotation=45)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.show()

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "cv_f1_macro": grid.best_score_,
            "test_accuracy": acc,
            "test_f1_macro": f1
        })

        # Track best overall model
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = best_model
            best_overall_name = name

    return results, best_overall_name, best_overall_model


def save_results_and_model(results, best_model_name, best_model):
    """
    Save the benchmark results to CSV and the best model to disk via joblib.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save benchmark table
    results_df = pd.DataFrame(results)
    results_path = os.path.join(MODELS_DIR, "model_benchmark.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved benchmark results to {results_path}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_model_name}) to {model_path}")


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    results, best_name, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("\n\n===== Summary of Results =====")
    for r in results:
        print(
            f"{r['model']}: "
            f"CV f1_macro={r['cv_f1_macro']:.4f}, "
            f"Test acc={r['test_accuracy']:.4f}, "
            f"Test f1_macro={r['test_f1_macro']:.4f}"
        )

    print(f"\nBest model on test (by macro F1): {best_name}")
    save_results_and_model(results, best_name, best_model)


if __name__ == "__main__":
    main()
