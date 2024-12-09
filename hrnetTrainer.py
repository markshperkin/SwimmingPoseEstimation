# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset
from pose_hrnet import get_pose_net
from loss import JointsMSELoss
import time
import yaml


class HRNetTrainer:
    def __init__(self,config_path="config.yaml", batch_size=8, learning_rate=1e-3, device="cuda"):
        """
        Initializes the HRNetTrainer class.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size


        # load configuration from YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # initialize dataset and dataloaders
        self.dataset = SwimmerDataset()
        print(f"Total number of frames in the dataset: {len(self.dataset)}")
        
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # load the HRNet model
        self.model = get_pose_net(self.config)
        self.model.to(self.device)

        # define the loss function and optimizer
        self.criterion = JointsMSELoss(use_target_weight=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs, save_path="hrnet_checkpoint.pth"):
        """
        Trains the HRNet model.
        """
        print(f"Training on device: {self.device}")

        # print dataset and dataloader information
        num_batches = len(self.dataloader)
        batch_size = self.dataloader.batch_size
        total_frames = len(self.dataloader.dataset)

        print(f"Training Information:")
        print(f"Total number of frames in the dataset: {total_frames}")
        print(f"Batch size: {batch_size}")
        print(f"Total number of batches: {num_batches}")

        # handle the case of an uneven last batch
        if total_frames % batch_size != 0:
            last_batch_size = total_frames % batch_size
            print(f"Last batch will have {last_batch_size} frames.")
        else:
            print("All batches are evenly sized.")

        # track the best loss for saving the best model
        best_loss = float("inf")
        total_training_time = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            epoch_start_time = time.time()

            for batch_idx, sample in enumerate(self.dataloader):
                # load data
                images = sample["image"].to(self.device)  # [B, C, H, W]
                target_heatmaps = sample["heatmaps"].to(self.device)  # [B, K, H, W]
                target_weight = sample["visibility"].to(self.device)  # [B, K]

                # forward pass
                self.optimizer.zero_grad()
                predicted_heatmaps = self.model(images)

                # loss computation
                loss = self.criterion(predicted_heatmaps, target_heatmaps, target_weight)

                # backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # compute epoch loss and log time
            epoch_loss = running_loss / len(self.dataloader)
            epoch_time = time.time() - epoch_start_time
            total_training_time += epoch_time

            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f}s, Average Loss: {epoch_loss:.6f}")

            # save the model if it achieves a lower loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at {save_path} with loss {best_loss:.6f}")

        # print total training time
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
                # load data
                images = sample["image"].to(self.device)  # [B, C, H, W]
                target_heatmaps = sample["heatmaps"].to(self.device)  # [B, K, H, W]
                target_weight = sample["visibility"].to(self.device)  # [B, K]

                # forward pass
                predicted_heatmaps = self.model(images)

                # compute loss
                loss = self.criterion(predicted_heatmaps, target_heatmaps, target_weight)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Evaluation Loss: {avg_loss:.6f}")
        return avg_loss

if __name__ == "__main__":
    # initialize the trainer
    trainer = HRNetTrainer(batch_size=8, learning_rate=1e-3, device="cuda")

    # train the model
    trainer.train(num_epochs=1)

    # evaluate the model
    trainer.evaluate()
