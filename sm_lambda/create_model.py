import json
import boto3
import os
from time import strftime

from sagemaker.pytorch.model import PyTorchModel

def lambda_handler(event, context):
    """
    모델 레지스트리에서 최신 버전의 모델 승인 상태를 변경하는 람다 함수.
    """
    
    try:
        sm_client = boto3.client("sagemaker")
        
        
        ## Custom Setting
        bucket = "yolov5-sagemaker-211217"
        model_package_group_name = "yolov5-detect-1640243744"
        role = "arn:aws:iam::687314952804:role/service-role/AmazonSageMaker-ExecutionRole-20211216T131915"
        instance_type = "ml.g4dn.xlarge"
        instance_count = 1
        detect_outputpath = f"s3://{bucket}/detect_result/output"
        manifest_obj = f"s3://{bucket}/detect_result/manifest/manifest.json"  ## Test data list
        
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
        

        try:
            sm_client.delete_model(ModelName=model_package_group_name)
        except:
            pass


        secret=get_secret()

        git_config = {'repo': 'https://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/yolov5',
                      'branch': 'master',
                      'username': secret['username'],
                      'password': secret['password']}        
        
        source_dir = 'yolo_v5/yolov5'

#         inf_image_uri = '687314952804.dkr.ecr.ap-northeast-2.amazonaws.com/yolov5:1.10.0-gpu-py38-inf'

        pytorch_model = PyTorchModel(model_data=model_data,
                                     role=role,
                                     image_uri=inf_image_uri,
                                     framework_version="1.10",
                                     py_version="py38",
                                     git_config=git_config,
                                     source_dir=source_dir,
                                     entry_point="detect.py")
        
        
        
        # then create transformer from PyTorchModel object
        transformer = pytorch_model.transformer(instance_count=instance_count, 
                                                instance_type=instance_type,
                                                output_path=detect_outputpath,
                                                strategy="SingleRecord",
                                                env={"DETECT" : "True", "s3_output": detect_outputpath})
        
        transformer.transform(
            data=manifest_obj,
            data_type="ManifestFile",
            content_type="application/x-image",
            wait=False,
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
        

# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:   
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
from botocore.exceptions import ClientError


def get_secret():

    secret_name = os.environ['sec_arn']
    print(f" ************** secret_name {secret_name}")
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