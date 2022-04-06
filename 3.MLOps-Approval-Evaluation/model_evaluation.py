import json
import boto3
import os
from time import strftime

from sagemaker.pytorch import PyTorch
from sagemaker.processing import Processor, ScriptProcessor, FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

import base64
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    """
    모델 레지스트리에서 최신 버전의 모델 승인 상태를 변경하는 람다 함수.
    """
    
    try:
        sm_client = boto3.client("sagemaker")
        
        
        ## Custom Setting
        bucket = os.environ["bucket_name"]  ### Revise the bucket name
        model_package_group_name = os.environ["model_package_group_name"] 
        role = os.environ["role"] 
        instance_type = os.environ["instanace_type"]
        instance_count = int(os.environ["instanace_count"])
        s3_test_path = f"s3://{bucket}/dataset/BCCD/test/images/"
        detect_output = f"s3://{bucket}/poc_yolov5/detect_output"
        code_location = f's3://{bucket}/poc_yolov5/sm_codes'
        
        experiment_name = 'yolov5-poc-exp1'
        ## SageMaker Experiments Setting
        try:
            sm_experiment = Experiment.load(experiment_name)
        except:
            sm_experiment = Experiment.create(
                experiment_name=experiment_name,
                tags=[{'Key': 'model-name', 'Value': 'yolov5'}]
            )    

        ## Trials Setting
        create_date = strftime("%m%d-%H%M%s")
        i_tag = instance_type.replace(".","-")
        trial = "-".join([i_tag,str(instance_count)])

        sm_trial = Trial.create(trial_name=f'{experiment_name}-{trial}-{create_date}',
                                experiment_name=experiment_name)

        job_name = f'{sm_trial.trial_name}'
            
        
        
        ##############################################
        # 람다 함수는 Event Bridge의 패턴 정보를 event 개체를 통해서 받습니다.
        ##############################################   
        print(f"event : {event}")
        try:
            model_package_arn = event['detail']["ModelPackageArn"]
            inf_image_uri = event['detail']["InferenceSpecification"]["Containers"][0]["Image"]
            model_data = event['detail']["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
            model_package_group_name = event['detail']["ModelPackageGroupName"]
        except:
            print("model_package_arn is not found in event")
            pass
        finally:
            response = sm_client.list_model_packages(
                    ModelPackageGroupName=model_package_group_name,
                    ModelApprovalStatus='Approved',
                    SortBy='CreationTime',
                    SortOrder='Descending' #|'Descending'
                )
            response = sm_client.describe_model_package(
                    ModelPackageName=response['ModelPackageSummaryList'][0]['ModelPackageArn']
                )
            model_package_arn = response["ModelPackageArn"]
            inf_image_uri = response["InferenceSpecification"]["Containers"][0]["Image"]
            model_data = response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

        print("model_package_arn: ", model_package_arn)
        print("inf_image_uri: ", inf_image_uri)  
        print("model_data: ", model_data)  
        print("model_package_group_name: ", model_package_group_name)
        
        secret=get_secret()
        
        codecommit_repo = os.environ["codecommit_repo"]

        git_config = {'repo':  codecommit_repo, ## 'https://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/yolov5',
                      'branch': 'main',
                      'username': secret['username'],
                      'password': secret['password']}        
        
        source_dir = 'yolov5'


        detect_processor = FrameworkProcessor(
            PyTorch,
            framework_version="1.9",
            role=role, 
            image_uri=inf_image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            code_location=code_location
            )
        
        
        detect_processor.run(
            code="detect.py",
            source_dir=source_dir,
            git_config=git_config,
            wait=False,
            inputs=[ProcessingInput(source=s3_test_path, input_name="test_data", destination="/opt/ml/processing/input"),
                    ProcessingInput(source=model_data.replace("model.tar.gz",""), input_name="model_weight", destination="/opt/ml/processing/weights")
            ],
            outputs=[
                ProcessingOutput(source="/opt/ml/processing/output", destination=detect_output),
            ],
            arguments=["--img", "640", "--conf", "0.25", "--source", "/opt/ml/processing/input", "--weights", "/opt/ml/processing/weights/model.tar.gz", "--project", "/opt/ml/processing/output"],
            job_name=job_name,
            experiment_config={
              'TrialName': job_name,
              'TrialComponentDisplayName': job_name,
            },
        )
        

        return_msg = f"Success"
        
        ##############################################        
        # 람다 함수의 리턴 정보를 구성하고 리턴 합니다.
        ##############################################        

        return {
            "statusCode": 200,
            "body": json.dumps(return_msg),
            "other_key": "example_value",
        }

    except BaseException as error:
        return_msg = f"There is no model_package_group_name {model_package_group_name}"                
        error_msg = f"An exception occurred: {error}"
        print(error_msg)    
        return {
            "statusCode": 500,
            "body": json.dumps(return_msg),
            "other_key": "example_value",
        }        
        

def get_secret():

    secret_name = os.environ['sec_arn']
    print(f" ************** secret_name {secret_name}")
    region_name = os.environ['region_name']
    secret = {}
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )


    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
        
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        secret = json.loads(secret)
    else:
        print("secret is not defined. Checking the Secrets Manager")

    return secret