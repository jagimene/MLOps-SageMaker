import sys
import time
import boto3
import json

sm_client = boto3.client('sagemaker')

def wait_for_training_jobs(estimators):
    statuses = ['Completed','Failed','Stopped']
    while True:
        finished = True        
        
        for estimator in estimators:        
            status = estimators[estimator].latest_training_job.describe()['TrainingJobStatus']
            finished *= status in statuses
            sys.stdout.write(".")
            
        if finished:
            sys.stdout.write("!\n")
            break
        else:
            time.sleep(10)
        
def wait_for_optmimization_jobs(tuners):
    statuses = ['Completed','Failed','Stopped']
    while True:
        finished = True        
        
        for tuner in tuners:
            status = tuners[tuner].describe()['HyperParameterTuningJobStatus']
            finished *= status in statuses
            sys.stdout.write(".")
            
        if finished:
            sys.stdout.write("!\n")
            break
        else:
            time.sleep(10)

def wait_for_transform_jobs(transformers):
    statuses = ['Completed','Failed','Stopped']
    while True:
        finished = True        
        
        for transform in transformers:
            job_name = transformers[transform].latest_transform_job.job_name
            status = sm_client.describe_transform_job(TransformJobName=job_name)['TransformJobStatus']
            finished *= status in statuses
            sys.stdout.write(".")
            
        if finished:
            sys.stdout.write("!\n")
            break
        else:
            time.sleep(10)
            
def create_or_update_iam_role(role_name, role_desc, asume_role_policy_document, policy_name, policy_document):
    iam = boto3.client('iam')
    
    response = None
    try:
        role_response = iam.get_role(RoleName=role_name)                 
        
        print('INFO: Role already exists, updating it...')
        role_arn = role_response['Role']['Arn']
        
        iam.update_role(RoleName=role_name, Description=role_desc)
                        
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(asume_role_policy_document))
                        
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document))
        
        print('INFO: Role updated: {}'.format(role_name))
        
        response = role_arn
        
    except iam.exceptions.NoSuchEntityException as e:
        print('INFO: Role does not exist, creating it...')
        try:
            create_role_response = iam.create_role(           
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(asume_role_policy_document),
                Description=role_desc)                     
            
            role_arn = create_role_response['Role']['Arn']

            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_document))
            
            print('INFO: Role created: {}'.format(role_arn))
            
            response = role_arn
            
        except Exception as e:
            print('ERROR: Failed to create role: {}'.format(role_name))
            print(e)
            
    except Exception as e:
        print('ERROR: Failed to update role: {}'.format(role_name))
        print(e)
        
    return response

def create_or_update_lambda_function(function_name, runtime, role_arn, handler,
                                     code, description, timeout, memory_size): 
    lambda_client = boto3.client('lambda')
    
    response = {None}
    try:
        response = lambda_client.get_function(FunctionName=function_name)  
        
        #Update function, because it was found. So, it does already exist
        response = lambda_client.update_function_configuration(FunctionName=function_name,
                                                               Runtime=runtime,
                                                               Role=role_arn,
                                                               Handler=handler,
                                                               Description=description,
                                                               Timeout=timeout,
                                                               MemorySize=memory_size)        
        
        print('Configuration update status: {}'.format(response['LastUpdateStatus']))
        
        response = lambda_client.update_function_code(FunctionName=function_name,               
                                                      S3Bucket=code['S3Bucket'],
                                                      S3Key=code['S3Key'])
        
        print('Code update status: {}'.format(response['LastUpdateStatus']))
        
    except lambda_client.exceptions.ResourceNotFoundException as e:
        try:
            #Create function, because it doesn't exist
            response = lambda_client.create_function(FunctionName=function_name,
                                                     Runtime=runtime,
                                                     Role=role_arn,
                                                     Handler=handler,
                                                     Code=code,
                                                     Description=description,
                                                     Timeout=timeout,
                                                     MemorySize=memory_size)

            print('Create status: {}'.format(response['LastUpdateStatus']))
        except Exception as e:
            print('Failed to create function: {}'.format(function_name))
            print(e)            
    except Exception as e:
        print('Failed to update function: {}'.format(function_name))
        print(e)   
        
    return response
