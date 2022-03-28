mkdir sagemaker-layer
cd sagemaker-layer
mkdir python
# Install the sagemaker modules in the python folder
pip install sagemaker --target ./python
# Remove tests and cache stuff (to reduce size)
find ./python -type d -name "tests" -exec rm -rfv {} +
find ./python -type d -name "__pycache__" -exec rm -rfv {} +

# Remove the python/numpy* folders since it will contain a numpy version for your host machine
rm -rf python/numpy*

# Download an AWS Linux compatible numpy package
# Navigate to https://pypi.org/project/numpy/#files.
# Search for and download newest *manylinux1_x86_64.whl package for your Python version (I have Python 3.7)
# curl "https://files.pythonhosted.org/packages/9b/04/c3846024ddc7514cde17087f62f0502abf85c53e8f69f6312c70db6d144e/numpy-1.19.2-cp37-cp37m-manylinux2010_x86_64.whl" -o "numpy-1.19.2-cp36-cp36m-manylinux1_x86_64.whl"
# unzip numpy-1.19.2-cp37-cp37m-manylinux2010_x86_64.whl -d python

zip -r sagemaker_lambda.zip .

# When zip file is ready, upload it to S3
# aws s3 cp sagemaker_lambda.zip s3://ai4iot-lambda/sagemaker_lambda_light.zip

# When upload is complete, goto Lambda layers to create a layer from the uploaded zip file.
