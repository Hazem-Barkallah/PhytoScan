#!/usr/bin/env bash
# Télécharge PlantVillage depuis Kaggle dans data/plantvillage.
# Prérequis : pip install kaggle, et ~/.kaggle/kaggle.json (jamais commité dans git).
set -euo pipefail
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/plantvillage --unzip
