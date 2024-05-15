import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as torchvision
from torchvision import transforms
from tqdm import tqdm

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config: TrainingConfig = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    def get_base_model(self):
        """Download the base model."""
        self.model = torch.load(self.config.updated_model_path).to(self.device)
        # NOTE: model.eval() for inference later on, not before training

    def get_dataloaders(self):
        """Create dataloaders for the training loop with the appropriate augmentations."""
        # TRAIN_DATA_PATH = "ImageFolder/images/train/"; self.config.training_data
        TRAIN_DATA_PATH = os.path.join(self.config.training_data, "train")
        VAL_DATA_PATH = os.path.join(self.config.training_data, "valid")
        TEST_DATA_PATH = os.path.join(self.config.training_data, "test")

        if self.config.params_augmentation:
            transform_img = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transform_img = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root=TRAIN_DATA_PATH, transform=transform_img
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=VAL_DATA_PATH, transform=transform_img
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA_PATH, transform=transform_img
        )

        # Dataloaders
        BATCH_SIZE = self.config.params_batch_size
        self.train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
        self.test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

    def save_checkpoint(self, epoch, optimizer, scheduler, val_accuracy):
        """Save the model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_accuracy": val_accuracy,
        }
        checkpoint_path = os.path.join(
            self.config.checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(state, checkpoint_path)
        print(
            f"Checkpoint saved at epoch {epoch} with validation accuracy: {val_accuracy}"
        )

    def load_checkpoint(self, checkpoint_path):
        """Load a model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return (
            checkpoint["epoch"],
            checkpoint["optimizer_state_dict"],
            checkpoint["scheduler_state_dict"],
            checkpoint["val_accuracy"],
        )

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.params_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        num_epochs = self.config.params_epochs

        best_val_accuracy = 0.0
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameters gradients
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}"
            )

            # validation step
            valid_accuracy = self.validate()

            if valid_accuracy > best_val_accuracy:
                best_val_accuracy = valid_accuracy
                self.save_checkpoint(epoch + 1, optimizer, scheduler, best_val_accuracy)

            # Step the scheduler
            scheduler.step()

    def validate(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy}%")
        return val_accuracy

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
        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy}%")
