import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def find_data_dir(data_root):
    chemins = glob.glob(os.path.join(data_root, "**", "color"), recursive=True)
    if not chemins:
        raise FileNotFoundError(
            f"Aucun dossier 'color' trouvé dans {data_root}. "
            "Lance d'abord : bash scripts/download_data.sh"
        )
    return chemins[0]


def build_dataframe(data_dir):
    lignes = []
    for classe in sorted(os.listdir(data_dir)):
        chemin_classe = os.path.join(data_dir, classe)
        if not os.path.isdir(chemin_classe):
            continue
        for fichier in os.listdir(chemin_classe):
            if fichier.lower().endswith((".jpg", ".jpeg", ".png")):
                chemin_fichier = os.path.join(chemin_classe, fichier)
                lignes.append((chemin_fichier, classe))  # tuples (chemin de fichier, nom de la classe)
    return pd.DataFrame(lignes, columns=["chemin", "classe"])


def prepare_dataframe(data, seed, n_par_classe=None):
    df = data.copy().sample(frac=1, random_state=seed)
    if n_par_classe is not None:
        df = df.groupby("classe").head(n_par_classe)  # n imgs au hasard par classe
    df = df.reset_index(drop=True)  # re-indexer les lignes après le shuffle

    # Chaque classe reçoit un numéro (c'est ce que le modèle apprend à prédire)
    classes = sorted(df["classe"].unique())
    df["label"] = df["classe"].map({c: i for i, c in enumerate(classes)})
    return df, classes


def split_dataframe(df, seed):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    return train_df, val_df, test_df


def charger(chemin, label, img_size):
    import tensorflow as tf

    img = tf.io.read_file(chemin)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    return img, label


def creer_dataset(donnees, img_size, batch_size, seed, melanger=False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((donnees["chemin"].values, donnees["label"].values))
    if melanger:
        ds = ds.shuffle(len(donnees), seed=seed)
    # Pas de .cache() : 54k images en float32 ne tiennent pas en RAM (~32 Go).
    return (
        ds.map(lambda c, l: charger(c, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_datasets(cfg):
    data_dir = find_data_dir(cfg["data_root"])
    data = build_dataframe(data_dir)
    df, classes = prepare_dataframe(data, cfg["seed"], cfg["n_par_classe"])
    print(df.shape[0], "images,", len(classes), "classes")

    train_df, val_df, test_df = split_dataframe(df, cfg["seed"])
    print("train :", len(train_df), "| validation :", len(val_df), "| test :", len(test_df))

    args = dict(img_size=cfg["img_size"], batch_size=cfg["batch_size"], seed=cfg["seed"])
    train_ds = creer_dataset(train_df, melanger=True, **args)
    val_ds = creer_dataset(val_df, **args)
    test_ds = creer_dataset(test_df, **args)
    return train_ds, val_ds, test_ds, test_df, classes
