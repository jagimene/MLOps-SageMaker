import argparse
import pickle
import os
import json
import tarfile
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve

def load_model(file, model_file='model.pkl'):
    if file.endswith('tar.gz'):
        with tarfile.open(file, 'r:gz') as tar:
            for name in tar.getnames():
                if name == model_file:
                    f = tar.extractfile(name)
                    return pickle.load(f)
            return None
    elif file.endswith('pkl'):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        return None

if __name__=='__main__':
    script_name = os.path.basename(__file__)
    print(f'INFO: {script_name}: Iniciando la evaluación de los modelos')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', type=str, required=True)
    parser.add_argument('--min-precision', type=float, required=True)    
    parser.add_argument('--test-data-file', type=str, required=True)
    parser.add_argument('--test-target-file', type=str, required=True)
    parser.add_argument('--thresholds-file', type=str, required=True)   
    parser.add_argument('--metrics-report-file', type=str, required=True)    
    
    args, _ = parser.parse_known_args()    
    
    print(f'INFO: {script_name}: Parámetros recibidos: {args}')
    
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'
    
    # Cargar datasets
    test_target_path = os.path.join(input_path, 'target', args.test_target_file)     
    test_target = pd.read_csv(test_target_path)
    
    test_data_path = os.path.join(input_path, 'data', args.test_data_file)     
    test_data = pd.read_csv(test_data_path)
    
    # Umbrales de decision por algoritmo
    algo_metrics = {'Algorithm':[], 'Threshold':[], 'Precision':[], 'Recall':[]}
    
    metrics_report = {}
    
    algos = args.algos.split(',')
    for algo in algos:
        model_path = os.path.join(input_path, algo, 'model.tar.gz')         

        # Carga modelo en memoria
        print(f'Cargando modelo: {model_path}')
        clf = load_model(model_path)
        
        # Obtiene predicciones con dataset para pruebas
        predictions = clf.predict_proba(test_data)[:, 1]
        
        # Busca umbral de decision
        precision, recall, thresholds = precision_recall_curve(test_target, predictions)
        operating_point_idx = np.argmax(precision>=args.min_precision)
        
        algo_metrics['Threshold'].append(thresholds[operating_point_idx])
        algo_metrics['Precision'].append(precision[operating_point_idx])
        algo_metrics['Recall'].append(recall[operating_point_idx])
        algo_metrics['Algorithm'].append(algo)
        
        metrics_report[algo] = {
            'precision': {'value': precision[operating_point_idx], 'standard_deviation': 'NaN'},
            'recall': {'value': recall[operating_point_idx], 'standard_deviation': 'NaN'}}
            
    
    # Guardar Thresholds    
    metrics = pd.DataFrame(algo_metrics)
    print(f'INFO: {script_name}: Thresholds encontrados')
    print(metrics)
    metrics.to_csv(os.path.join(output_path, args.thresholds_file), index=False)    
    
    # Guardar reporte de metricas para cada modelo
    for algo in metrics_report:
        with open(os.path.join(output_path, f'{algo}_metrics.json'), 'w') as f:
            json.dump({'binary_classification_metrics':metrics_report[algo]},f)        

    
    print(f'INFO: {script_name}: Finalizando la evaluación de los modelos')
