# Plantilla de preprocesamiento de datos

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def read_and_preprocessing(file_path):
    dataset = pd.read_csv(file_path)
    dataset = replace_null(dataset)
    dataset = encode_categorical_data(dataset)
    return dataset


def replace_null(dataset):
    if dataset.isnull().values.any():
        print_message("Valores nulos. Se reemplazarán con la media de la columna.")
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(dataset.iloc[:, 1:3])
        dataset.iloc[:, 1:3] = imputer.transform(dataset.iloc[:, 1:3])
    else:
        print("El dataset no contiene valores nulos.")
    return dataset


def print_message(message):
    print(message)
    print()


def encode_categorical_data(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X = one_hot_encoding(X, [0])
    y = label_encoding(y)

    dataset = pd.DataFrame(X)
    dataset["Purchased"] = y

    return dataset


def one_hot_encoding(dataset, category_indices):
    ct = ColumnTransformer(
        transformers=[
            ("one_hot_encoder", OneHotEncoder(categories="auto"), category_indices)
        ],
        remainder="passthrough",
    )
    dataset = ct.fit_transform(dataset)
    return dataset


def label_encoding(dataset):
    labelEncoder = LabelEncoder()
    dataset = labelEncoder.fit_transform(dataset)
    return dataset


# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
def divide_dataset(dataset):
    independent_matrix = dataset.iloc[:, :-1].values
    target_vector = dataset.iloc[:, -1].values

    # Dividimos los datos: 80% para entrenamiento y 20% para prueba
    # random_state asegura que los resultados sean reproducibles
    return train_test_split(
        independent_matrix, target_vector, test_size=0.2, random_state=0
    )


# Escalado de características, se aplica una estandarización en este caso
def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


if __name__ == "__main__":
    file_path = "Data.csv"
    dataset = read_and_preprocessing(file_path)
    X_train, X_test, y_train, y_test = divide_dataset(dataset)
    X_train_standardized, X_test_standardized = feature_scaling(X_train, X_test)

    print("Datos de X_train estandarizados:")
    print(X_train_standardized)
    print()
    print("Datos de X_test estandarizados:")
    print(X_test_standardized)

    # Para este caso, nuestro algoritmo es de clasificación (queremos predecir si un cliente comprará o no un producto)
    # Por lo tanto, no es necesario estandarizar y_train y y_test
    # ** Cuando se trate de un caso de predicción, como en la regresión, entonces si vamos a estandarizar y_train y y_test
    print()
    print("Datos de y_train:")
    print(y_train)
    print()
    print("Datos de y_test:")
    print(y_test)
    print()
