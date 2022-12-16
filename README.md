![Logo Henry](_src/logo-henry-white-lg.png)

# PI02-DTS05 DATATHON

## Introducción

En el marco del segundo proyecto individual de la carrera de Data Science en Henry se propuso una competencia (Datathon) la cual consitió en realizar un modelo de ML capaz de predecir si la cantidad de dias de internación de un paciente superará los 8 dias.

Para ello se suministró un dataset de train (utilizado para entrenar al modelo) y uno de test (utilizado para predecir la cantidad de dias). Inicialmente se realiza un analisis exploratorio de datos (EDA) para disernir que features son relevantes para realizar las predicciones. Luego, se reliza la seleción de los modelos de ML y se los entrena con los datos proporcionados. Las predicciones se enviaron a un dashboard que se diponibilizó por el equipo de labs de Henry, donde se puntuó al modelo con las metricas de `recall` y `acurracy`.

## Instalación

El primer paso es clonar el repositorio, puede hacerse utilizando el comando:

```cmd
git clone https://github.com/agusdm97/PI02-DTS05.git
```

Se requiere de la creación y activación de un entorno virtual de python3.10.5; luego, de la instalación de las dependencias ubicadas en el archivo requirements.txt con el siguiente comando:

```cmd
pip install -r requirements.txt
```

Posteriormente se elige el modelo de ML a utilizar, en la carpeta `models` se encuentran notebooks con siguientes modelos:

- Random Forest Classifier
- Boosting Classifier
- Bagging (Decission Tree Classifier)

        Nota importante: El entrenamiento de estos modelos pueden llevar una gran cantidad de poder de computo, la solución ideal es guardar el modelo entrenado. Por cuestiones de tiempo no se pudo llevar a cabo dicha solución en este proyecto.

Finalmente los resultados de las predicciones se guardan automaticamente en la carpeta `resultados` con las especifiaciones indicadas en la consigna del proyecto.
