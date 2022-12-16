from typing import Any
import pandas as pd
import datetime as dt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


def transform_df(path: str, test: bool = False) -> pd.DataFrame:
    """
    transform_df
    -------
    Construye un DataFrame a partir de los datos unbicados en el archivo csv ubicado en

    `path` y luego realiza las transformaciónes necesarias para trabajar con el modelo de ML.

    Parameters
    ----------
    `path` : str
        ruta del archivo csv.
    `test` : bool, optional
        indica si el archivo es el de test, por defecto False.

    Returns
    -------
    `pandas.DataFrame`
        dataframe con todas las transformaciones necesarias.
    """

    df = pd.read_csv(filepath_or_buffer=path)

    if not test:
        df["stay_labed"] = (df["Stay (in days)"] > 8) * 1
        df.drop(columns=["Stay (in days)"], inplace=True)

    df = pd.concat([df, pd.get_dummies(data=df["gender"], prefix="gender")], axis=1)
    df.drop(columns=["gender"], inplace=True)

    age_encoder = LabelEncoder()
    df["age_labed"] = age_encoder.fit_transform(df["Age"].values.reshape(-1, 1))
    df.drop(columns=["Age"], inplace=True)

    insurance_encoder = LabelEncoder()
    df["insurance_labed"] = insurance_encoder.fit_transform(
        df["Insurance"].values.reshape(-1, 1)
    )
    df.drop(columns=["Insurance"], inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(data=df["Severity of Illness"], prefix="severity")], axis=1
    )
    df.drop(columns=["Severity of Illness"], inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(data=df["health_conditions"], prefix="health_cond")], axis=1
    )
    df.drop(columns=["health_conditions"], inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(data=df["Type of Admission"], prefix="admission")], axis=1
    )
    df.drop(columns=["Type of Admission"], inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(data=df["Department"], prefix="department")], axis=1
    )
    df.drop(columns=["Department"], inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(data=df["doctor_name"], prefix="doctor")], axis=1
    )
    df.drop(columns=["doctor_name"], inplace=True)

    admission_scaler = StandardScaler()
    df["admission_deposit_scaled"] = admission_scaler.fit_transform(
        df["Admission_Deposit"].values.reshape(-1, 1)
    )
    df.drop(columns=["Admission_Deposit"], inplace=True)

    df.drop(
        columns=["Ward_Facility_Code", "patientid", "Visitors with Patient"],
        inplace=True,
    )

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    return df


def save_results(series: pd.Series) -> None:
    """
    save_results
    ------------
    Recibe una serie de pandas con las predicciones del modelo y las guarda en un

    archivo .csv con las especificaciones del proyectoen la carpeta resultados

    asignandole un numero de prueba y la fecha

    Parameters
    ----------
    series : pd.Series
        Serie de pandas con las predicciones del modelo
    """

    os.chdir("C:/Users/Agustín/Desktop/Code/PI02-DTS05/resultados")
    directories = os.listdir()

    tz = dt.timezone(offset=dt.timedelta(hours=-3))

    if len(directories) == 0:
        num = 1
    else:
        list_ = []
        for dir in directories:
            list_.append(dir.split(" ")[2])
        num = int(max(list_)) + 1

    dir_name = f"Prueba N° {num} - {dt.datetime.now(tz=tz).strftime(r'%d %b %H-%M')}"
    os.mkdir(dir_name)
    os.chdir(os.path.join(os.getcwd(), dir_name))

    series.to_csv(
        path_or_buf="./agusdm97.csv",
        index=False,
        header=["pred"],
    )
