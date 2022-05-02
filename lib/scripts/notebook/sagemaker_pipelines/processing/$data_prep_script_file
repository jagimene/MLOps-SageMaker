import argparse
import pickle
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def to_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        
if __name__=='__main__':
    script_name = os.path.basename(__file__)
    
    print(f'INFO: {script_name}: Iniciando la preparación de los datos')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--data-file', type=str, default='train.csv')
    parser.add_argument('--train-data-file', type=str)
    parser.add_argument('--train-target-file', type=str)
    parser.add_argument('--test-data-file', type=str)
    parser.add_argument('--test-target-file', type=str)
    parser.add_argument('--encoder-file', type=str)
    
    args, _ = parser.parse_known_args()    
    
    print(f'INFO: {script_name}: Parámetros recibidos: {args}')
    
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'
    
    data_path = os.path.join(input_path, args.data_file) 
    
    # Cargar dataset
    data = pd.read_csv(data_path)
    
    # Eliminar caracteres especiales y reemplazar espacios por guiones bajos
    data.columns = [''.join (c if c.isalnum() else '_' for c in str(column)) for column in data.columns]
    
    # Selección de columnas
    columns = ['State', 'Account_Length', 'Area_Code', 'Int_l_Plan','VMail_Plan', 'VMail_Message', 
           'Day_Mins', 'Day_Calls','Eve_Mins', 'Eve_Calls', 'Night_Mins', 'Night_Calls', 
           'Intl_Mins', 'Intl_Calls', 'CustServ_Calls', 'Churn_']
    data = data[columns]
    
    # Eliminación del . al final de la palabra False o True en la columna Churn_ y renombrarla a Churn
    data['Churn_']=data['Churn_'].str.replace('.','')
    data.rename(columns={'Churn_':'Churn'}, inplace=True)
    
    # One hot encoding de variables categóricas
    columns = ['State','Area_Code']
    encoder = OneHotEncoder().fit(data[columns])
    
    transformed = encoder.transform(data[columns]).toarray()
    
    data.drop(columns,axis=1, inplace=True)
    data = pd.concat([data,pd.DataFrame(transformed, columns=encoder.get_feature_names())],axis=1)
    
    # Reemplazar yes/no por 1/0 en columnas Int_l_Plan y VMail_Plan
    data['Int_l_Plan'] = data['Int_l_Plan'].map(dict(yes=1, no=0))
    data['VMail_Plan'] = data['VMail_Plan'].map(dict(yes=1, no=0))
    
    # Reemplazar True/False por 1/0 en columna Churn
    data['Churn'] = data['Churn'].map({'True': 1, 'False': 0})
    
    # Separar la etiqueta o target del resto de los datos
    target = data[['Churn']]
    data.drop(['Churn'], axis=1, inplace=True)
    
    # Y dividimos en train (80%) y test (20%), manteniendo las mismas proporciones de observaciones por cada clase
    train_data, test_data, train_target, test_target = train_test_split(data, target, stratify=target, 
                                                                        test_size=args.test_size)
    
    print('Train: {0} records with clasess: 0={1[0]}% and 1={1[1]}%'.format(train_target.shape[0],
                                             round(train_target['Churn'].value_counts(normalize=True) * 100, 1)))

    print('Test: {0} records with clasess: 0={1[0]}% and 1={1[1]}%'.format(test_target.shape[0],
                                             round(test_target['Churn'].value_counts(normalize=True) * 100, 1)))
    
    # Guardar los dataframes resultantes y el encoder
    train_data.to_csv(os.path.join(output_path, 'train_data', args.train_data_file), index=False)
    train_target.to_csv(os.path.join(output_path, 'train_target', args.train_target_file), index=False)
    test_data.to_csv(os.path.join(output_path, 'test_data', args.test_data_file), index=False)
    test_target.to_csv(os.path.join(output_path, 'test_target', args.test_target_file), index=False)
    to_pkl(encoder, os.path.join(output_path, 'encoder', args.encoder_file))
    
    print(f'INFO: {script_name}: Finalizando la preparación de los datos')
