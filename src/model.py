import keras
from keras import layers


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
    ])


def build_transfer_model(cfg):
    h, w = cfg["img_size"]
    base = keras.applications.MobileNetV2(input_shape=(h, w, 3), include_top=False, weights=cfg["weights"])
    base.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(h, w, 3)),
        build_augmentation(),
        layers.Rescaling(1 / 127.5, offset=-1),  # pixels de [0, 255] vers [-1, 1]
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(cfg["num_classes"], activation="softmax"),
    ])
    compile_model(model, lr=1e-3)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_for_finetuning(model, base, n_couches=30, lr=1e-5):
    base.trainable = True
    for couche in base.layers[:-n_couches]:
        couche.trainable = False
    for couche in base.layers[-n_couches:]:
        if isinstance(couche, layers.BatchNormalization):
            couche.trainable = False  # on garde les BatchNorm gelées (plus stable)
    compile_model(model, lr=lr)


def build_cnn(cfg):
    h, w = cfg["img_size"]
    cnn = keras.Sequential([
        layers.Input(shape=(h, w, 3)),
        build_augmentation(),
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(cfg["num_classes"], activation="softmax"),
    ])
    compile_model(cnn, lr=1e-3)
    return cnn
