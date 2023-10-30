from sklearnex import patch_sklearn
patch_sklearn()

import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

def obtener_parametros():
    n_neighbors=3,
    algorithm='brute',
    p=1,
    weights='distance',
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--dataset_path", type=str, help="File path to training data")
    parser.add_argument("--min_samples_split", type=int, help="Min # of samples split")
    parser.add_argument("--n_neighbors", type=int, help="Number of neighbors to vote")
    parser.add_argument("--algorithm", type=str, help="Algorithm: ball_tree or auto")
    parser.add_argument("--p", type=int, help="Function to compute the distance: 1 (manhattan_distance) or 2 (euclidean_distance)")
    parser.add_argument("--weights", type=str, help="Weighting of votes: uniforme (same value) or distance (value based on distance)")
    parser.add_argument("--show_params", type=bool, help="Show the params inserted")

    args = parser.parse_args()
    return args

def inicialiazar_mlflow():
    mlflow.start_run()
    mlflow.sklearn.autolog()

def imprimir_parametros_ingresados(args):
    if not args.show_params:
        return None

    parametros  = [
        f"Data file: {args.dataset_path}",
        f"Min Samples split: {args.min_samples_split}",
        f"N Neighbors: {args.n_neighbors}",
        f"Algorithm: {args.algorithm}",
        f"P: {args.p}",
        f"Weights: {args.weights}",
    ]

    print("Parametros ingresados:")
    for parametros in parametros:
        print(parametros)

def mostrar_parametros_en_logs(args):
    mlflow.log_param('Data file', str(args.dataset_path))
    mlflow.log_param('Min Samples split', str(args.min_samples_split))
    mlflow.log_param('N Neighbors', str(args.n_neighbors))
    mlflow.log_param('Algorithm', str(args.algorithm))
    mlflow.log_param('P', str(args.p))
    mlflow.log_param('Weights', str(args.weights))
    
def obtener_X_y_de_dataset(args):
    data = pd.read_csv(args.dataset_path)

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]

    return X,y

def evaluar_modelo(KN_class, X_test, y_test):
    y_pred = KN_class.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    return f1

def main():
    args = obtener_parametros()

    inicialiazar_mlflow()
    mostrar_parametros_en_logs(args)

    imprimir_parametros_ingresados(args)
    
    X,y = obtener_X_y_de_dataset(args)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Create the NNN model
    KN_class = KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                    algorithm=args.algorithm,
                                    p=args.p,
                                    weights=args.weights,
                                    n_jobs=-1)
    KN_class.fit(X_train, y_train)

    f1 = evaluar_modelo(KN_class, X_test, y_test)

    # imprimir metrica en mlflow
    mlflow.log_metric('F1 Score', float(f1))

    registered_model_name="sklearn-K-Nearest-Neighbors"

    print("Registrando el modelo via MLFlow...")
    mlflow.sklearn.log_model(
        sk_model=KN_class,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    print("Guardando el modelo via MLFlow...")
    mlflow.sklearn.save_model(
        sk_model=KN_class,
        path=os.path.join(registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == '__main__':
    main()
