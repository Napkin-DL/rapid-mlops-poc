import json
import boto3
import os


sagemaker       = boto3.client('sagemaker')
cfn_client      = boto3.client('cloudformation')


def lambda_handler(event, context):
    stage = event['stage']

    if stage == 'Training':
        training_job_name = event['training_job_name']
        training_details = describe_training_job(training_job_name)
        print(training_details)

        status = training_details['TrainingJobStatus']
        if status == 'Completed':
            s3_output_path = training_details['OutputDataConfig']['S3OutputPath']
            model_data_url = os.path.join(s3_output_path, training_details['TrainingJobName'], 'output/model.tar.gz')

            event['message'] = 'Training job "{}" complete. Model data uploaded to "{}"'.format(training_job_name, model_data_url)
            event['model_data_url'] = model_data_url
            event['training_job'] = training_details['TrainingJobName']
        elif status == 'Failed':
            failure_reason = training_details['FailureReason']
            event['message'] = 'Training job failed. {}'.format(failure_reason)
    
    event['status'] = status
    
    print(event)
    
    return event

def describe_training_job(name):
    """ Describe SageMaker training job identified by input name.
    Args:
        name (string): Name of SageMaker training job to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the training job.
    """
    try:
        response = sagemaker.describe_training_job(
            TrainingJobName = name
        )
    except Exception as e:
        print(e)
        print('Unable to describe training job.')
        raise(e)
    
    return response
