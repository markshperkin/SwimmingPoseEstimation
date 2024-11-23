from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset  # Ensure the path is correct
from pose_hrnet import get_pose_net  # Ensure the path is correct
from loss import JointsMSELoss
import os
import time


class HRNetTrainer:
    def __init__(self, batch_size=8, learning_rate=1e-3, device="cuda"):
        """
        Initializes the HRNetTrainer class.

        Args:
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # Configuration embedded directly in the class
        self.config = {
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

        # Initialize dataset and dataloaders
        self.dataset = SwimmerDataset()  # Uses default root_dir='data'
        print(f"Total number of frames in the dataset: {len(self.dataset)}")
        
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Load the HRNet model
        self.model = get_pose_net(self.config)
        self.model.to(self.device)

        # Define the loss function and optimizer
        self.criterion = JointsMSELoss(use_target_weight=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs, save_path="hrnet_checkpoint.pth"):
        """
        Trains the HRNet model.

        Args:
            num_epochs (int): Number of epochs to train for.
            save_path (str): Path to save the model checkpoints.
        """
        print(f"Training on device: {self.device}")

        # Print dataset and dataloader information
        num_batches = len(self.dataloader)  # Total number of batches
        batch_size = self.dataloader.batch_size  # Frames per batch
        total_frames = len(self.dataloader.dataset)  # Total frames in dataset

        print(f"Training Information:")
        print(f"Total number of frames in the dataset: {total_frames}")
        print(f"Batch size: {batch_size}")
        print(f"Total number of batches: {num_batches}")

        # Handle the case of an uneven last batch
        if total_frames % batch_size != 0:
            last_batch_size = total_frames % batch_size
            print(f"Last batch will have {last_batch_size} frames.")
        else:
            print("All batches are evenly sized.")

        # Track the best loss for saving the best model
        best_loss = float("inf")
        total_training_time = 0  # Initialize total training time

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            epoch_start_time = time.time()

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

            # Compute epoch loss and log time
            epoch_loss = running_loss / len(self.dataloader)
            epoch_time = time.time() - epoch_start_time
            total_training_time += epoch_time

            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f}s, Average Loss: {epoch_loss:.6f}")

            # Save the model if it achieves a lower loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at {save_path} with loss {best_loss:.6f}")

        # Print total training time
        print(f"Total training time: {total_training_time:.2f}s")
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
        """
        Visualizes the predictions of the HRNet model.
        """
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
    # Initialize the trainer
    trainer = HRNetTrainer(batch_size=8, learning_rate=1e-3, device="cuda")

    # Train the model
    trainer.train(num_epochs=100)  # Specify number of epochs here

    # Evaluate the model
    trainer.evaluate()
