"""
Usage :
    python -m src.predict chemin/vers/feuille.jpg
"""
import argparse
import json
import os

import keras
import numpy as np

from src.config import load_config


def predict_image(model, classes, image_path, img_size, top_k=3):
    img = keras.utils.load_img(image_path, target_size=img_size)
    x = np.expand_dims(keras.utils.img_to_array(img), axis=0)
    proba = model.predict(x, verbose=0)[0]
    return [(classes[i], float(proba[i])) for i in proba.argsort()[::-1][:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Prédit la maladie d'une feuille.")
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or os.path.join(cfg["model_dir"], "modele_plantes.keras")
    model = keras.models.load_model(model_path)
    with open(os.path.join(cfg["model_dir"], "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)

    print("Les 3 prédictions les plus probables :")
    for classe, p in predict_image(model, classes, args.image, cfg["img_size"]):
        print(f"  {classe} : {p:.1%}")


if __name__ == "__main__":
    main()
