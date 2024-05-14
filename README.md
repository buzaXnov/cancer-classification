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

Q: Freezing layers means that frozen layers will not be trained? 
Q: Shouldn't my last layer be a softmax activation function? 

TODO: 
1. Implement quantized transfer learning after the entire project is completed: 
https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html

2. Trying playing around with the last layers, the numbers and sizes of them. 