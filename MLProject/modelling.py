import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
import mlflow
import mlflow.sklearn


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_churn_preprocessed.csv")


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def main(n_estimators: int, max_depth):
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    mlflow.set_experiment("telco-churn-ci")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Manual logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close(cm_fig)

        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        # Save run_id so the CI workflow can reference it for Docker build
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Run ID: {run_id}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=lambda x: None if x == "None" else int(x), default=10)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
