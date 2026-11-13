import yaml


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["img_size"] = tuple(cfg["img_size"])
    return cfg
