from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from dataloader import SwimmerDataset  # Ensure the path is correct
from pose_hrnet import get_pose_net  # Ensure the path is correct
from loss import JointsMSELoss, JointsOHKMMSELoss # Import the loss class
import os
import time


class HRNetTrainer:
    def __init__(self, config_path, dataset_path, batch_size=8, learning_rate=1e-3, device="cuda"):
        """
        Initializes the HRNetTrainer class.

        Args:
            config_path (str): Path to the configuration YAML file.
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # Load the configuration file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize the dataset and dataloaders
        self.dataset = SwimmerDataset(dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Load the HRNet model
        self.model = get_pose_net(self.config)
        self.model.to(self.device)

        # Define the loss function and optimizer
        self.criterion = JointsMSELoss(use_target_weight=True)  # Use the imported JointsMSELoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs=1000, save_path="hrnet_checkpoint.pth"):
        """
        Trains the HRNet model.

        Args:
            num_epochs (int): Number of epochs to train for.
            save_path (str): Path to save the model checkpoints.
        """
        print(f"Training on device: {self.device}")

        # Track the best loss for saving the best model
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            start_time = time.time()

            for batch_idx, sample in enumerate(self.dataloader):
                # Load data
                images = sample["image"].to(self.device)  # [B, C, H, W]
                target_heatmaps = sample["heatmaps"].to(self.device)  # [B, K, H, W]
                target_weight = sample["visibility"].to(self.device)  # [B, K]

                # Forward pass
                self.optimizer.zero_grad()
                predicted_heatmaps = self.model(images)

                # Loss computation
                loss = self.criterion(predicted_heatmaps, target_heatmaps, target_weight)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Print batch progress
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(self.dataloader)}], "
                          f"Loss: {loss.item():.6f}")

            epoch_loss = running_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - start_time:.2f}s, "
                  f"Average Loss: {epoch_loss:.6f}")

            # Save the model if it achieves a lower loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at {save_path} with loss {best_loss:.6f}")

        print("Training complete!")

    def evaluate(self):
        """
        Evaluates the HRNet model on the training dataset.

        Returns:
            float: Average loss on the training dataset.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for sample in self.dataloader:
                # Load data
                images = sample["image"].to(self.device)  # [B, C, H, W]
                target_heatmaps = sample["heatmaps"].to(self.device)  # [B, K, H, W]
                target_weight = sample["visibility"].to(self.device)  # [B, K]

                # Forward pass
                predicted_heatmaps = self.model(images)

                # Compute loss
                loss = self.criterion(predicted_heatmaps, target_heatmaps, target_weight)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Evaluation Loss: {avg_loss:.6f}")
        return avg_loss

def visualize_predictions(self, num_samples=5):
    self.model.eval()
    samples = iter(self.dataloader)

    with torch.no_grad():
        for _ in range(num_samples):
            sample = next(samples)
            image = sample["image"][0].to(self.device)
            target_heatmaps = sample["heatmaps"][0].cpu().numpy()
            predicted_heatmaps = self.model(image.unsqueeze(0))[0].cpu().numpy()

            # Display the image, ground truth, and predicted heatmaps
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0).cpu().numpy())
            plt.title("Input Image")

            plt.subplot(1, 3, 2)
            plt.imshow(target_heatmaps.sum(axis=0))
            plt.title("Ground Truth Heatmaps")

            plt.subplot(1, 3, 3)
            plt.imshow(predicted_heatmaps.sum(axis=0))
            plt.title("Predicted Heatmaps")

            plt.show()


if __name__ == "__main__":
    # Paths to config.yaml and dataset
    config_path = "config.yaml"
    dataset_path = "annotations/annotations/person_keypoints_Train.json"

    # Ensure the configuration file exists
    if not os.path.exists(config_path):
        config = {
            'MODEL': {
                'EXTRA': {
                    'STAGE2': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'NUM_BLOCKS': [4, 4],
                        'NUM_CHANNELS': [32, 64],
                        'BLOCK': 'BASIC',
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 3,
                        'NUM_BLOCKS': [4, 4, 4],
                        'NUM_CHANNELS': [32, 64, 128],
                        'BLOCK': 'BASIC',
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 4,
                        'NUM_BLOCKS': [4, 4, 4, 4],
                        'NUM_CHANNELS': [32, 64, 128, 256],
                        'BLOCK': 'BASIC',
                        'FUSE_METHOD': 'SUM'
                    },
                    'FINAL_CONV_KERNEL': 1,
                    'PRETRAINED_LAYERS': ['*']
                },
                'NUM_JOINTS': 13,
                'INIT_WEIGHTS': True
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    # Initialize the trainer
    trainer = HRNetTrainer(config_path, dataset_path, batch_size=8, learning_rate=1e-3, device="cuda")

    # Train the model
    trainer.train(num_epochs=1000, save_path="hrnet_checkpoint.pth")

    # Evaluate the model
    trainer.evaluate()
