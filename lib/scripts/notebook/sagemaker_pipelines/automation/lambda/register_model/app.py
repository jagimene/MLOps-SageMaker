import boto3

sm_client = boto3.client('sagemaker')

def get_parameter(key, event):
    parameter_value = None
    if (key in event):
        parameter_value = event[key]

    else:
        raise KeyError('{} key not found in function input!'.format(key)+
                      ' The input received was: {}.'.format(event))
        
    return parameter_value
    

#Retrieve training job name from event and create a model.
def lambda_handler(event, context):
    response = None
    
    print(event)
    
    job_name = get_parameter('TrainingJobName', event)
    model_name = get_parameter('ModelName', event)
    code_path = get_parameter('CodePath', event)
        
    try:
        
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        model_data = response['ModelArtifacts']['S3ModelArtifacts']
        container = response['AlgorithmSpecification']['TrainingImage']
        role_arn = response['RoleArn']
        
        print("Training job: {} has artifacts at: {}.".format(job_name, model_data))
        
        try:
            response = sm_client.create_model(ModelName=model_name,
                                   PrimaryContainer={
                                       'Image':container,
                                       'Mode':'SingleModel',
                                       'ModelDataUrl':model_data,
                                       'Environment':{
                                           'SAGEMAKER_PROGRAM': 'train_and_deploy.py',
                                           'SAGEMAKER_SUBMIT_DIRECTORY': code_path
                                       }
                                   },
                                   ExecutionRoleArn=role_arn
            )

            print(response)

        except Exception as e:
            response = ('Failed to create model')
            print(e)
            print('{} Attempted to create a model for job name: {}.'.format(response, job_name))

    except Exception as e:
        response = ('Failed to read training job artifacts!'+ 
                    ' The training job may not exist or the job name may be incorrect.'+ 
                    ' Check SageMaker to confirm the job name.')
        print(e)
        print('{} Attempted to read job name: {}.'.format(response, job_name))
            
    return {
        'statusCode': 200,
        'ModelArn': response['ModelArn'],
        'TrainingJobName': job_name
    }
