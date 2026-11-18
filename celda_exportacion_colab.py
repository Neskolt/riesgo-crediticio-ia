# === CELDA DE EXPORTACIÓN PARA GOOGLE COLAB ===
# Ejecutar DESPUÉS de entrenar modelo_controlado.

import json
import joblib
import sklearn
from pathlib import Path
from google.colab import files

feature_order = list(X.columns)

joblib.dump(modelo_controlado, "modelo_crediticio.joblib")

with open("model_features.json", "w", encoding="utf-8") as f:
    json.dump(feature_order, f, ensure_ascii=False, indent=2)

with open("version_sklearn.txt", "w", encoding="utf-8") as f:
    f.write(sklearn.__version__)

print("Modelo exportado correctamente.")
print("Versión de scikit-learn usada en Colab:", sklearn.__version__)
print("Número de variables:", len(feature_order))
print("Orden de variables:", feature_order)

files.download("modelo_crediticio.joblib")
files.download("model_features.json")
files.download("version_sklearn.txt")
