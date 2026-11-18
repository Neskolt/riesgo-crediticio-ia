# Evaluación de Riesgo Crediticio

Aplicación Streamlit para desplegar el Árbol de Decisión controlado desarrollado en Google Colab.

## Archivos

- `app.py`: aplicación web.
- `modelo_crediticio.joblib`: modelo entrenado. **Debes exportarlo desde Colab.**
- `model_features.json`: orden exacto de variables usado durante el entrenamiento.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: configuración visual.
- `celda_exportacion_colab.py`: código que debes copiar a una nueva celda al final del notebook.

## Paso 1 — Exportar el modelo

1. Abre el notebook final en Google Colab.
2. Ejecuta `Ejecutar todas`.
3. Copia el contenido de `celda_exportacion_colab.py` a una nueva celda al final.
4. Ejecuta esa celda.
5. Se descargarán:
   - `modelo_crediticio.joblib`
   - `model_features.json`
   - `version_sklearn.txt`
6. Copia los dos primeros archivos a esta carpeta.
7. Revisa `version_sklearn.txt`. Antes del despliegue fijaremos esa misma versión en `requirements.txt`.

## Paso 2 — Prueba local (opcional)

En una terminal dentro de la carpeta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Paso 3 — Despliegue

Cuando la app esté validada:
1. Crear repositorio GitHub.
2. Subir los archivos del proyecto.
3. Conectar el repositorio a Streamlit Community Cloud.
4. Seleccionar `app.py`.
5. Publicar y probar la URL desde PC y teléfono.

## Nota metodológica

La interfaz mantiene las 23 variables del modelo entrenado. Se organizan por relevancia y comprensión, pero no se eliminan variables porque eso cambiaría la estructura de entrada esperada por el modelo.


## Versión final validada

El modelo incluido fue exportado desde Google Colab con:

- scikit-learn: **1.6.1**
- 23 variables de entrada
- `DecisionTreeClassifier`
- `max_depth=5`
- `min_samples_leaf=50`
- `class_weight="balanced"`
- `random_state=42`

`requirements.txt` fija `scikit-learn==1.6.1` para mantener compatibilidad con el archivo serializado.
