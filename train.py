import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class HRNetTrainer:
    """
    A class to handle training of the HRNetKeypointDetector.
    """
    def __init__(self, model, dataloader, device='cuda', lr=1e-4):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module): The HRNet model.
            dataloader (DataLoader): DataLoader for the dataset.
            device (str): Device to train the model on ('cuda' or 'cpu').
            lr (float): Learning rate for the optimizer.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()  # Mean Squared Error for heatmaps
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)  # Optional
        self.epoch_losses = []  # Store losses for each epoch

    def train(self, num_epochs=10, save_path="hrnet_model.pth"):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            save_path (str): Path to save the trained model.

        Returns:
            None
        """
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.dataloader):
                # Move data to the device
                images = batch['image'].to(self.device)  # [B, C, H, W]
                target_heatmaps = batch['heatmaps'].to(self.device)  # [B, K, H, W]

                # Forward pass
                self.optimizer.zero_grad()
                predicted_heatmaps = self.model(images)

                # Compute loss
                loss = self.criterion(predicted_heatmaps, target_heatmaps)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

                # Log batch progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            # Log epoch loss
            epoch_loss /= len(self.dataloader)
            self.epoch_losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}")

            # Step the scheduler
            self.scheduler.step()

            # Save the model checkpoint after each epoch
            torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")

        print("Training complete. Model saved to:", save_path)

    def evaluate(self, dataloader):
        """
        Evaluate the model on a validation/test dataset.

        Args:
            dataloader (DataLoader): DataLoader for the validation or test dataset.

        Returns:
            float: Average validation/test loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                target_heatmaps = batch['heatmaps'].to(self.device)

                predicted_heatmaps = self.model(images)
                loss = self.criterion(predicted_heatmaps, target_heatmaps)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Validation/Test Loss: {avg_loss:.4f}")
        return avg_loss
