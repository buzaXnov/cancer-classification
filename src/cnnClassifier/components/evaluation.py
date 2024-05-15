import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import torch
import torch.utils.data as data
import torchvision as torchvision
from torchvision import transforms
from tqdm import tqdm

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config) -> None:
        self.config: EvaluationConfig = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    def get_trained_model(self):
        """Download the base model."""
        self.model = torch.load(self.config.path_of_model).to(self.device)

    def get_dataloader(self):
        """Create the dataloader for the test dataset."""
        TEST_DATA_PATH = os.path.join(self.config.training_data, "test")

        transform_img = transforms.Compose(
            [
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA_PATH, transform=transform_img
        )

        # Dataloaders
        BATCH_SIZE = self.config.params_batch_size
        self.test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

    def test(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.test_accuracy = 100 * correct / total

        print(f"Test Accuracy: {self.test_accuracy}%")

    def save_score(self):
        scores = {"test_accuracy": self.test_accuracy}
        save_json(scores, Path("scores.json"))

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"test_accuracy": self.test_accuracy})

            logger.info("Params and metrics logged.")

            if tracking_uri_type_store != "file":
                mlflow.pytorch.log_model(
                    self.model, "model", registered_model_name="VGG16"
                )
            else:
                mlflow.pytorch.log_model(self.model, "model")

            logger.info("Model logged.")
