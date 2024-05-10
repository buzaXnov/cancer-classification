import torch
import torch.nn as nn
import torchvision

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config: PrepareBaseModelConfig = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: torchvision.models.vgg.VGG

    def get_base_model(self):
        """Download the base model."""
        # TODO: try using vgg16_bn (batch norm)
        self.model = torchvision.models.vgg16(weights=self.config.params_weights)

        torch.save(self.model, self.config.base_model_path)

    def update_base_model(self):
        """Replaces the last layer with a custom num_classes sized layer."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Adding the new last layer
        self.model.classifier[-1] = nn.Linear(
            in_features=4096, out_features=self.config.params_classes
        )

        print(self.model)

        torch.save(self.model, self.config.updated_base_model_path)
