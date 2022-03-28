import json
import boto3
import os
from time import strftime
import subprocess
import sagemaker

import datetime
import glob
import os
import time
import warnings

from sagemaker.pytorch import PyTorch

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial


instance_type = os.environ["INSTANCE_TYPE"]


def lambda_handler(event, context):
    
    role = "arn:aws:iam::687314952804:role/service-role/AmazonSageMaker-ExecutionRole-20211216T131915"
    sagemaker_session = sagemaker.Session()
    
    experiment_name = 'yolov5-poc-exp1'
    
#     instance_type = 'ml.p3.2xlarge'  # 'ml.p3.16xlarge', 'ml.p3dn.24xlarge', 'ml.p4d.24xlarge', 'local_gpu'
    # instance_type = 'local_gpu'
    instance_count = 1
    do_spot_training = False
    max_wait = None
    max_run = 1*60*60
    
    try:
        sm_experiment = Experiment.load(experiment_name)
    except:
        # if exist, add timestamp to differentiate
        sm_experiment = Experiment.create(experiment_name=experiment_name.format(datetime.datetime.now().strftime("%H%M%S")),
                                          tags=[
                                              {
                                                  'Key': 'model-name',
                                                  'Value': 'yolov5'
                                              }
                                          ])    
    
    
    create_date = strftime("%m%d-%H%M%s")    
    spot = 's' if do_spot_training else 'd'
    i_tag = instance_type.replace(".","-")
    trial = "-".join([i_tag,str(instance_count),spot])
       
    sm_trial = Trial.create(trial_name=f'{experiment_name}-{trial}-{create_date}',
                            experiment_name=experiment_name)

    job_name = f'{sm_trial.trial_name}'

    
    bucket = 'yolov5-sagemaker-211217'
    code_location = f's3://{bucket}/sm_codes'
    output_path = f's3://{bucket}/poc_yolov5/output' 
    s3_log_path = f's3://{bucket}/tf_logs'
    
    hyperparameters = {
        'data': 'data_sm.yaml',
        'cfg': 'yolov5s.yaml',
        'weights': 'weights/yolov5s.pt',
        'batch-size': 128,
#         'epochs': 10,
        'epochs': 3,
        'project': '/opt/ml/model',
        'workers': 8,
        'freeze': 10
    }
    
    
    s3_data_path = f's3://{bucket}/dataset/BCCD'
    checkpoint_s3_bucket = f's3://{bucket}/checkpoints'
    
    image_uri = '687314952804.dkr.ecr.ap-northeast-2.amazonaws.com/yolov5:1.10.0-gpu-py38-tv'
    distribution = {}

    if hyperparameters.get('sagemakerdp') and hyperparameters['sagemakerdp']:
        train_job_name = 'smdp-dist'
        distribution["smdistributed"]={ 
                            "dataparallel": {
                                "enabled": True
                            }
                    }

    else:
        distribution["mpi"]={
                            "enabled": True,
        #                     "processes_per_host": 8, # Pick your processes_per_host
        #                     "custom_mpi_options": "-verbose -x orte_base_help_aggregate=0 "
                      }

    if do_spot_training:
        max_wait = max_run

        
    source_dir = 'yolov5'
    
    secret=get_secret()
    
    git_config = {'repo': 'https://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/yolov5',
                  'branch': 'master',
                  'username': secret['username'],
                  'password': secret['password']}
    
    source_dir = 'yolo_v5/yolov5'
    
    # all input configurations, parameters, and metrics specified in estimator 
    # definition are automatically tracked
    estimator = PyTorch(
        entry_point='train_sm.py',
        source_dir=source_dir,
        git_config=git_config,
        role=role,
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=1024,
        code_location = code_location,
        output_path=output_path,
        hyperparameters=hyperparameters,
        distribution=distribution,
        disable_profiler=True,
        debugger_hook_config=False,
        max_run=max_run,
        use_spot_instances=do_spot_training,  # spot instance 활용
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_bucket,
        TrainingInputMode='File', ## FastFile
    #     max_retry_attempts=0
    )
    
    # Now associate the estimator with the Experiment and Trial
    estimator.fit(
        inputs={'yolov5_input': s3_data_path},
        job_name=job_name,
        wait=False,
    )
    
    event['training_job_name'] = job_name
    event['stage'] = 'Training'
    
    return event
    
# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:   
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
from botocore.exceptions import ClientError


def get_secret():

    secret_name = "arn:aws:secretsmanager:ap-northeast-2:687314952804:secret:codecommit-cred-qv4SVU"
    region_name = "ap-northeast-2"
    secret = {}
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
        
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        secret = json.loads(secret)
    else:
        print("secret is not defined. Checking the Secrets Manager")

    return secret

