
import json
import boto3
import botocore
import os


sm_client = boto3.client("sagemaker")

model_package_group_name = os.environ['MODEL_PACKAGE_GROUP_NAME']
model_package_group_desc = os.environ['MODEL_PACKAGE_GROUP_DESC']

training_image = '366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3'


def lambda_handler(event, context):
    
    modelpackage_inference_specification =  {
        "InferenceSpecification": {
          "Containers": [
             {
                "Image": training_image,
             }
          ],
          "SupportedContentTypes": [ "application/x-image" ],
          "SupportedResponseMIMETypes": [ "application/x-image" ],
        }
    }
     
    model_data_url = event['model_data_url'] 
    
    
    # Specify the model data
    modelpackage_inference_specification["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]=model_data_url
    
    create_model_package_input_dict = {
        "ModelPackageGroupName" : model_package_group_name,
        "ModelPackageDescription" : model_package_group_desc,
        "ModelApprovalStatus" : "PendingManualApproval"
    }

    create_model_package_input_dict.update(modelpackage_inference_specification)
    modelpackage_inference_specification["InferenceSpecification"]["Containers"][0]
    
    try:
        create_mode_package_response = sm_client.create_model_package(**create_model_package_input_dict)
    except botocore.exceptions.ClientError as ce:
        # When model package group does not exit
        print('Model package grop does not exist. Creating a new one')
        if ce.operation_name == "CreateModelPackage":
            if ce.response["Error"]["Message"] == "Model Package Group does not exist.":
                # Create model package group
                create_model_package_group_response = sm_client.create_model_package_group(
                    ModelPackageGroupName=model_package_group_name,
                    ModelPackageGroupDescription=model_package_group_desc,
                )
                
                create_mode_package_response = sm_client.create_model_package(**create_model_package_input_dict)
                
    return event
