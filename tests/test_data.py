import pandas as pd

from src.data import prepare_dataframe, split_dataframe


def fake_data(n_classes=4, n_par_classe=40):
    lignes = [(f"img_{c}_{i}.jpg", f"classe_{c}") for c in range(n_classes) for i in range(n_par_classe)]
    return pd.DataFrame(lignes, columns=["chemin", "classe"])


def test_n_par_classe_limite_les_images():
    df, classes = prepare_dataframe(fake_data(), seed=42, n_par_classe=10)
    assert len(df) == 40
    assert (df["classe"].value_counts() == 10).all()
    assert len(classes) == 4


def test_none_garde_tout():
    df, _ = prepare_dataframe(fake_data(), seed=42, n_par_classe=None)
    assert len(df) == 160


def test_labels_correspondent_aux_classes_triees():
    df, classes = prepare_dataframe(fake_data(), seed=42)
    assert classes == sorted(classes)
    assert (df["label"] == df["classe"].map(classes.index)).all()


def test_split_proportions_et_sans_chevauchement():
    df, _ = prepare_dataframe(fake_data(), seed=42)
    train, val, test = split_dataframe(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train["chemin"]) & set(test["chemin"])
    assert not set(train["chemin"]) & set(val["chemin"])
    assert not set(val["chemin"]) & set(test["chemin"])
    assert abs(len(train) / len(df) - 0.70) < 0.02


def test_split_stratifie():
    df, _ = prepare_dataframe(fake_data(), seed=42)
    _, _, test = split_dataframe(df, seed=42)
    assert test["label"].value_counts().nunique() == 1  # même nb d'images par classe
