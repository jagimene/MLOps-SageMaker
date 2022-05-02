import argparse
import pickle
import os
import io
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Carga el modelo en memoria
def model_fn(model_dir):
    print('Cargando modelo: model_fn')
    clf = read_pkl(os.path.join(model_dir, "model.pkl"))
    return clf

# Deserealiza el body de la petición para poder generar las predicciones
def input_fn(request_body, request_content_type):

    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        input_data = pd.DataFrame.from_dict(input_data)
        # TODO: Es importante asegurarse de que las columnas se encuentran en el orden adecuado
        return input_data
        
    elif request_content_type == 'text/csv':      
        input_data = io.StringIO(request_body)        
        return pd.read_csv(input_data, header=None)
    else:
        raise ValueError("El endpoint del modelo solamente soporta Content-Types: 'application/json' o 'text/csv' como entrada")
                
# Genera la predicción sobre el objeto deserializado, con el modelo previamente cargado en memoria
def predict_fn(input_data, model):
    predict_proba = getattr(model, 'predict_proba', None)
    if callable(predict_proba):
        return predict_proba(input_data)[:, 1]
    else:
        return model.predict(input_data)

# Serializa el resultado de la predicción al correspondiente content type deseado
def output_fn(predictions, response_content_type):
    if response_content_type == 'application/json':        
        return json.dumps(predictions.tolist())
    elif response_content_type == 'text/csv':
        predictions_response = io.StringIO()
        np.savetxt(predictions_response, predictions, delimiter=',')
        return predictions_response.getvalue()
    else:
        raise ValueError("El endpoint del modelo solamente soporta Content-Types: 'application/json' o 'text/csv' como respuesta")        
        
def read_pkl(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def to_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def random_forest(**hyperparameters):
    return RandomForestClassifier(n_jobs=-1, 
                                  min_samples_split=hyperparameters['min_samples_split'],
                                  n_estimators=hyperparameters['n_estimators'],
                                  max_depth=hyperparameters['max_depth'],
                                  max_features=hyperparameters['max_features'])

def gradient_boosting(**hyperparameters):
    return GradientBoostingClassifier(learning_rate=hyperparameters['learning_rate'],
                                      min_samples_split=hyperparameters['min_samples_split'],
                                      n_estimators=hyperparameters['n_estimators'],
                                      max_depth=hyperparameters['max_depth'],
                                      max_features=hyperparameters['max_features'])

def extra_trees(**hyperparameters):
    return ExtraTreesClassifier(n_jobs=-1, 
                                min_samples_split=hyperparameters['min_samples_split'],
                                n_estimators=hyperparameters['n_estimators'],
                                max_depth=hyperparameters['max_depth'],
                                max_features=hyperparameters['max_features'])

def invalid_algorithm(**hyperparameters):
    raise Exception('Invalid Algorithm')
    
def algorithm_selector(algorithm, **hyperparameters):
    algorithms = {
        'RandomForest': random_forest,
        'GradientBoosting': gradient_boosting,
        'ExtraTrees': extra_trees
    }
    
    clf = algorithms.get(algorithm, invalid_algorithm)    
    return clf(**hyperparameters)


if __name__=='__main__':
    script_name = os.path.basename(__file__)
    print(f'INFO: {script_name}: Iniciando entrenamiento del modelo')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_DATA'))
    parser.add_argument('--train-target', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_TARGET'))
    
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--target-metric', type=str)
    
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--min-samples-split', type=int)
    parser.add_argument('--n-estimators', type=int)
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--max-features', type=int)
    
            
    args, _ = parser.parse_known_args()
    
    print(f'INFO: {script_name}: Parametros recibidos: {args}')

    # Cargar datasets
    files = os.listdir(args.train_data)
    if len(files) == 1:
        train_data = pd.read_csv(os.path.join(args.train_data, files[0]))
    else:
        raise Exception('Mas de un archivo recibido para el channel Data')
    
    files = os.listdir(args.train_target)
    if len(files) == 1:
        train_target = pd.read_csv(os.path.join(args.train_target, files[0]))
        train_target = train_target['Churn'].tolist()
    else:
        raise Exception('Mas de un archivo recibido para el channel Target')
     
    clf = algorithm_selector(args.algorithm, 
                             learning_rate=args.learning_rate,
                             min_samples_split=args.min_samples_split,
                             n_estimators=args.n_estimators,
                             max_depth=args.max_depth,
                             max_features=args.max_features)
    
    skf = StratifiedKFold(n_splits=args.splits)    
    cv_scores = cross_validate(clf, train_data, train_target, cv=skf, scoring=args.target_metric, n_jobs=-1)
    print('{} = {}%'.format(args.target_metric, cv_scores['test_score'].mean().round(4)*100))
    
    # Entrenar el modelo
    clf.fit(train_data, train_target) 
    
    # Guardar modelo
    to_pkl(clf, os.path.join(args.model_dir, 'model.pkl'))

    print(f'INFO: {script_name}: Finalizando el entrenamiento del modelo')   

