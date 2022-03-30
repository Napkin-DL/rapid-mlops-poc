ACCOUNT_ID=$1
REGION=$2
ALGO_NAME=$3
FULL_NAME=$4


docker build -f Dockerfile -t $FULL_NAME .

$(aws ecr get-login --no-include-email --registry-ids $ACCOUNT_ID)

aws ecr describe-repositories --repository-names $ALGO_NAME || aws ecr create-repository --repository-name $ALGO_NAME

docker push $FULL_NAME