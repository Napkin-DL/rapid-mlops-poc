
import json
import boto3
import tarfile
from io import BytesIO
import os
import pickle
from io import StringIO
import csv


s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
s3_resource = boto3.resource('s3')


acc_col_num = os.environ['ACC_COL_NUM']


def lambda_handler(event, context):
    # print(event)    
    model_data_url = event['model_data_url']
    bucket = event['bucket']
    key_value = model_data_url.split(bucket)[1][1:]
    print(key_value)
    tar_file_obj = s3.get_object(Bucket=bucket, Key=key_value)
    tar_content = tar_file_obj ['Body'].read()
    
    accuracy = 0
    
    with tarfile.open(fileobj = BytesIO(tar_content)) as tar:
      for tar_resource in tar:
          if (tar_resource.isfile()):
            # if "txt" in tar_resource.name:
            #     inner_file_bytes = tar.extractfile(tar_resource).read()
            #     print(inner_file_bytes)
            #     accuracy = inner_file_bytes.decode('utf-8')
            if "results.csv" in tar_resource.name:
                inner_file_bytes = tar.extractfile(tar_resource).read()
                file_data = inner_file_bytes.decode('utf-8')
                file = StringIO(file_data)
                csv_data = csv.reader(file, delimiter=",")
                
                max_line = len(list(csv_data))
                
                file = StringIO(file_data)
                csv_data = csv.reader(file, delimiter=",")
                
                line_count = 0
                
                for row in csv_data:
                    line_count += 1
                    if line_count == max_line:
                        accuracy = row[int(acc_col_num)].lstrip()
                        
    print("accuracy is " + accuracy)
    
    desired_accuracy = event['desired_accuracy']
    
    if accuracy > desired_accuracy:
        event['train_result'] = "PASS"
        print("PASS")
    else:
        event['train_result'] = "FAIL"
        print("FAIL")

    return event
