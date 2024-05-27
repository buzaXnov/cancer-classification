# cancer-classification
End-to-end cancer classification using MLflow, DVC and PyTorch. 

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

This workflow is the same for every step of the project:
1. Data ingestion
2. Model preparation
3. Training and testing loop

Use the research folder and the appropriate notebook for creating different files where you will later paste the functions
and classes once you have created them and tested them. 


## Model creation
You need to remove the last layers, FC layers is using VGG16 and insert my own custom FC layers with 4 neurons representing 4 classes.
Link to PyTorch model: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html


Since you shouldn't be posting your password and tracking username for mlflow, you should look into secrets.yaml file. 

TODO: Make a separate notebook for exploring the dataset. 

Q: Freezing layers means that frozen layers will not be trained? 
Q: Shouldn't my last layer be a softmax activation function? 
Q: Why do we use Dagshub for mlflow? 

TODO: 
1. Implement quantized transfer learning after the entire project is completed: 
https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html

2. Trying playing around with the last layers, the numbers and sizes of them. (implement vgg16_bn model?)
Define the augmentations and right image size for my model and dataset. 

3. Compare VGG16, InceptionV3 Models and EfficientNetB0, but not before you get a satisfactory test accuracy on VGG16.
I.e. complete the project, and then create a notebook for data exploration and augmentation. 

4. Put the best models in the models/ folder in the project root directory. 


### Streamlit (frontend)
[streamlit] (https://streamlit.io)

Picking a model: 
```python
model = st.radio("Pick a model", models) # to allow the user to check results from various CNN models
```

Data visualization:
visualize the metrics such as accuracy, precision, recall and the confusion matrix for every loaded model

Insert a compare all models metrics section? 

### mlflow experiment tracking with dahshub
[dagshub](https://dagshub.com)

MLFLOW_TRACKING_URI=https://dagshub.com/buzaXnov/cancer-classification.mlflow \
MLFLOW_TRACKING_USERNAME=buzaXnov \
MLFLOW_TRACKING_PASSWORD=c2f97b6e1897763abc2b78d3781adbbf30ba2026 \
python script.py

Run this to export as env variables:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/buzaXnov/cancer-classification.mlflow

export MLFLOW_TRACKING_USERNAME=buzaXnov

export MLFLOW_TRACKING_PASSWORD=c2f97b6e1897763abc2b78d3781adbbf30ba2026
```

Worth reading: http://pytorch.org/docs/master/notes/autograd.html



TODO: uncomment log into mlflow method in stage 04 model eval


DVC CLI commands:
dvc init

dvc repro

dvc dag - to check a visual representation of mutual dependencies

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 527673296413.dkr.ecr.us-east-1.amazonaws.com/chest

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

