"""
Usage :
    python -m src.train                    # MobileNetV2 + CNN
    python -m src.train --model transfer   # seulement MobileNetV2
    python -m src.train --model cnn        # seulement le CNN
"""
import argparse
import json
import os

import keras
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import load_datasets
from src.model import build_cnn, build_transfer_model, unfreeze_for_finetuning


def plot_history(hist, titre, path=None):
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="validation")
    plt.title(titre)
    plt.xlabel("Époque")
    plt.legend()
    if path:
        plt.savefig(path, dpi=150)
    plt.close()


def make_callbacks(patience_stop, patience_lr):
    """EarlyStopping (restaure les meilleurs poids) + ReduceLROnPlateau (lr / 2 si val_loss stagne)."""
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_stop, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience_lr, min_lr=1e-7),
    ]


def train_transfer(cfg, train_ds, val_ds):
    model, base = build_transfer_model(cfg)
    print("Paramètres :", f"{model.count_params():,}")
    callbacks = make_callbacks(patience_stop=3, patience_lr=1)

    # Étape 1 : on entraîne uniquement la nouvelle tête (base gelée)
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_tete"], callbacks=callbacks, verbose=2)
    plot_history(hist1, "Accuracy - étape 1", os.path.join(cfg["figures_dir"], "accuracy_etape1.png"))

    # Étape 2 : fine-tuning des 30 dernières couches avec un petit learning rate
    unfreeze_for_finetuning(model, base, n_couches=30, lr=1e-5)
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_finetuning"], callbacks=callbacks, verbose=2)
    plot_history(hist2, "Accuracy - fine-tuning", os.path.join(cfg["figures_dir"], "accuracy_finetuning.png"))
    return model


def train_cnn(cfg, train_ds, val_ds):
    cnn = build_cnn(cfg)
    callbacks = make_callbacks(patience_stop=5, patience_lr=2)
    cnn.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_cnn"], callbacks=callbacks, verbose=2)
    return cnn


def main():
    parser = argparse.ArgumentParser(description="Entraîne les modèles de classification de maladies de plantes.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["transfer", "cnn", "both"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    keras.utils.set_random_seed(cfg["seed"])
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["figures_dir"], exist_ok=True)

    train_ds, val_ds, _, _, classes = load_datasets(cfg)
    with open(os.path.join(cfg["model_dir"], "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    if args.model in ("transfer", "both"):
        model = train_transfer(cfg, train_ds, val_ds)
        model.save(os.path.join(cfg["model_dir"], "modele_plantes.keras"))
    if args.model in ("cnn", "both"):
        cnn = train_cnn(cfg, train_ds, val_ds)
        cnn.save(os.path.join(cfg["model_dir"], "cnn_scratch.keras"))


if __name__ == "__main__":
    main()
