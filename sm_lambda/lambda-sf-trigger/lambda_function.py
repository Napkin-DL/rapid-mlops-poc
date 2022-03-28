import json
import boto3
import os


s3 = boto3.client('s3')
sf = boto3.client('stepfunctions')


state_machine_arn = os.environ['STATE_MACHINE_ARN']


def lambda_handler(event, context):
    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    print('Reading {} from {}'.format(file_key, bucket_name))
    
    obj = s3.get_object(Bucket = bucket_name, Key = file_key)
    
    file_content = obj['Body'].read().decode('utf-8')
    
    json_content = json.loads(file_content)
    print(json_content)
    
    sf.start_execution(
        stateMachineArn = state_machine_arn,
        input = json.dumps(json_content))
    
    return event
