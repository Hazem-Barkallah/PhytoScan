"""Évaluation : métriques, rapport par classe, matrice de confusion, erreurs fréquentes."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def predict_test(model, test_ds, test_df):
    """Retourne (y_true, y_pred) sur le jeu de test."""
    probas = model.predict(test_ds, verbose=0)
    y_pred = probas.argmax(axis=1)  # classe prédite = la plus probable
    y_true = test_df["label"].values  # vraie classe
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def per_class_report(y_true, y_pred, classes):
    return classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes, zero_division=0
    )


def plot_confusion_matrix(y_true, y_pred, classes, path=None):
    n = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(n), classes, rotation=90, fontsize=7)
    plt.yticks(range(n), classes, fontsize=7)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    return cm


def top_errors(y_true, y_pred, classes, n=10):
    resultats = pd.DataFrame({
        "vraie classe": [classes[i] for i in y_true],
        "classe prédite": [classes[i] for i in y_pred],
    })
    erreurs = resultats[resultats["vraie classe"] != resultats["classe prédite"]]
    return len(erreurs), erreurs.value_counts().head(n)


def plot_misclassified(test_df, y_true, y_pred, classes, n=8, path=None):
    from PIL import Image

    idx_erreurs = np.where(y_pred != y_true)[0]
    if len(idx_erreurs) == 0:
        print("Aucune erreur sur le jeu de test.")
        return
    plt.figure(figsize=(14, 7))
    for k, i in enumerate(idx_erreurs[:n]):
        plt.subplot(2, 4, k + 1)
        plt.imshow(Image.open(test_df["chemin"].iloc[i]).convert("RGB"))
        plt.title(f"vrai : {classes[y_true[i]]}\nprédit : {classes[y_pred[i]]}", fontsize=8, color="red")
        plt.axis("off")
    plt.subplots_adjust(top=0.85, hspace=0.5)
    if path:
        plt.savefig(path, dpi=150)


def main():
    import keras

    from src.config import load_config
    from src.data import load_datasets

    parser = argparse.ArgumentParser(description="Évalue un modèle sauvegardé sur le jeu de test.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="Chemin du modèle MobileNetV2 (.keras)")
    parser.add_argument("--cnn", default=None, help="Chemin du CNN scratch (.keras), pour le tableau comparatif")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    model_path = args.model or os.path.join(cfg["model_dir"], "modele_plantes.keras")

    _, _, test_ds, test_df, classes = load_datasets(cfg)

    model = keras.models.load_model(model_path)
    y_true, y_pred = predict_test(model, test_ds, test_df)
    acc, f1 = compute_metrics(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1:.4f}\n")
    print(per_class_report(y_true, y_pred, classes))

    plot_confusion_matrix(y_true, y_pred, classes, os.path.join(cfg["figures_dir"], "confusion_matrix.png"))
    n_err, paires = top_errors(y_true, y_pred, classes)
    print(n_err, "erreurs sur", len(y_true), "images de test\n")
    print(paires)
    plot_misclassified(test_df, y_true, y_pred, classes, path=os.path.join(cfg["figures_dir"], "misclassified.png"))

    if args.cnn:
        cnn = keras.models.load_model(args.cnn)
        _, y_pred_cnn = predict_test(cnn, test_ds, test_df)
        acc_cnn, f1_cnn = compute_metrics(y_true, y_pred_cnn)
        tableau = pd.DataFrame({
            "Modèle": ["CNN simple (from scratch)", "MobileNetV2 (transfer learning)"],
            "Accuracy": [acc_cnn, acc],
            "F1 macro": [f1_cnn, f1],
        })
        print("\n", tableau.to_string(index=False))
        tableau.to_csv(os.path.join(cfg["figures_dir"], "comparaison.csv"), index=False)

    with open(os.path.join(cfg["figures_dir"], "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1}, f, indent=2)


if __name__ == "__main__":
    main()
