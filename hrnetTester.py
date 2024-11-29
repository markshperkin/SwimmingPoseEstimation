import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset  # Ensure this is the correct path
from pose_hrnet import get_pose_net  # Ensure this is the correct path to HRNet implementation


class HRNetTester:
    def __init__(self, config_path, batch_size=1, device="cuda"):
        """
        Initializes the HRNetTester class.

        Args:
            config_path (str): Path to the configuration YAML file.
            batch_size (int): Batch size for the dataloader.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        print("Initializing HRNetTester...")  # Debug print
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # Load the configuration file
        print(f"Loading configuration from {config_path}...")  # Debug print
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize the dataset and dataloader
        print("Initializing dataset and dataloader...")  # Debug print
        self.dataset = SwimmerDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Load the HRNet model
        print("Loading HRNet model...")  # Debug print
        self.model = get_pose_net(self.config)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def load_weights(self, weights_path):
        """
        Load a set of weights into the HRNet model.

        Args:
            weights_path (str): Path to the weights file.
        """
        print(f"Loading weights from: {weights_path}...")  # Debug print
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        print("Weights successfully loaded!")  # Debug print

    def test_and_visualize(self, frame_index=None):
        """
        Tests the HRNet model on a single batch or a specific frame and visualizes the results.

        Args:
            frame_index (int, optional): Index of the specific frame to test. If None, loads the next batch.
        """
        if frame_index is not None:
            print(f"Testing on frame index: {frame_index}...")  # Debug print
            sample = self.dataset[frame_index]  # Access a specific frame by index
            image = sample["image"].unsqueeze(0).to(self.device)  # [1, C, H, W]
            ground_truth_heatmaps = sample["heatmaps"].unsqueeze(0)  # [1, K, H, W]
            keypoints_gt = sample["keypoints"].cpu().numpy()  # Ground truth keypoints
            meta = sample["meta"]  # Metadata for debugging purposes
        else:
            print("Loading sample batch for testing...")  # Debug print
            sample = next(iter(self.dataloader))
            image = sample["image"].to(self.device)  # [B, C, H, W]
            ground_truth_heatmaps = sample["heatmaps"]  # [B, K, H, W]
            keypoints_gt = sample["keypoints"][0].cpu().numpy()  # Ground truth keypoints
            meta = sample["meta"]  # Metadata for debugging purposes

        print(f"Ground Truth Keypoints (Original Scale): {keypoints_gt}")  # Debug print
        print(f"Image Dimensions: {image[0].shape[1:]}")  # Debug print

        # Forward pass through the model
        print("Performing forward pass through the model...")  # Debug print
        with torch.no_grad():
            predicted_heatmaps = self.model(image)  # Output shape: [B, K, H, W]

        # Extract predicted keypoints
        batch_idx = 0  # Display the first image in the batch
        num_keypoints = predicted_heatmaps.shape[1]
        pred_keypoints = []
        confidences = []

        print("Extracting predicted keypoints and confidences...")  # Debug print
        for k in range(num_keypoints):
            heatmap = predicted_heatmaps[batch_idx, k].cpu().numpy()
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidences.append(heatmap[max_pos])
            pred_keypoints.append(max_pos[::-1])  # (x, y)

        pred_keypoints = np.array(pred_keypoints)
        confidences = np.array(confidences)
        print(f"Predicted Keypoints (Heatmap Scale): {pred_keypoints}")  # Debug print
        print(f"Confidences: {confidences}")  # Debug print

        # Rescale keypoints
        original_size = (image.shape[3], image.shape[2])  # Width, Height
        pred_keypoints_rescaled = pred_keypoints * np.array([original_size[0] / heatmap.shape[1], original_size[1] / heatmap.shape[0]])
        print(f"Rescaled Predicted Keypoints: {pred_keypoints_rescaled}")  # Debug print

        # Calculate distances
        print("Calculating distances between GT and Predicted Keypoints...")  # Debug print
        distances = self.calculate_distance(keypoints_gt, pred_keypoints_rescaled, original_size)
        print(f"Distances: {distances}")  # Debug print

        # Visualize the results (same visualization code as before)
        fig, axs = plt.subplots(4, num_keypoints + 1, figsize=(30, 20))
        fig.suptitle("Original Image, GT Keypoints, Confidence, Predicted Keypoints, and Distance", fontsize=16)

        # Display the original image with keypoints
        axs[0, 0].imshow(image[batch_idx].permute(1, 2, 0).cpu().numpy())
        axs[0, 0].set_title("Image with Keypoints")
        axs[0, 0].axis("off")
        for kp in keypoints_gt:
            axs[0, 0].plot(kp[0], kp[1], "ro")  # Ground truth in red

        # Display ground truth keypoints
        for i in range(num_keypoints):
            axs[0, i + 1].imshow(ground_truth_heatmaps[batch_idx, i].cpu().numpy(), cmap="hot")
            axs[0, i + 1].set_title(f"GT Keypoint {i + 1}")
            axs[0, i + 1].axis("off")

        # Display confidence bar graph
        for i, confidence in enumerate(confidences):
            axs[1, i + 1].bar([0], [confidence], color="blue")
            axs[1, i + 1].set_ylim(0, 1)
            axs[1, i + 1].set_title(f"Confidence")
            axs[1, i + 1].set_xticks([])

        # Display predicted keypoints
        for i in range(num_keypoints):
            axs[2, i + 1].imshow(predicted_heatmaps[batch_idx, i].cpu().numpy(), cmap="hot")
            axs[2, i + 1].set_title(f"Pred Keypoint {i + 1}")
            axs[2, i + 1].axis("off")

        # Display distance bar graph
        for i, distance in enumerate(distances):
            axs[3, i + 1].bar([0], [distance], color="green")
            axs[3, i + 1].set_ylim(0, max(distances) + 1)
            axs[3, i + 1].set_title(f"Distance")
            axs[3, i + 1].set_xticks([])

        axs[3, 0].axis("off")  # Empty space for alignment
        axs[2, 0].axis("off")
        axs[1, 0].axis("off")

        plt.tight_layout()
        plt.show()

    def calculate_distance(self, gt_keypoints, pred_keypoints, original_size):
        """
        Calculate the Euclidean distance between ground truth and predicted keypoints.

        Args:
            gt_keypoints (np.array): Ground truth keypoints, shape [num_keypoints, 2].
            pred_keypoints (np.array): Predicted keypoints, shape [num_keypoints, 2].
            original_size (tuple): Original image size (width, height).

        Returns:
            distances (list): Euclidean distances for each keypoint.
        """
        print("Inside calculate_distance function...")  # Debug print
        distances = []
        for gt, pred in zip(gt_keypoints, pred_keypoints):
            # Debugging: Print GT and Predicted Keypoints
            print(f"GT Keypoint: {gt}, Predicted Keypoint: {pred}")  # Debug print

            # Compute Euclidean distance
            distance = np.linalg.norm(pred - gt)
            distances.append(distance)

        return distances

if __name__ == "__main__":
    # Path to config.yaml and weights
    config_path = "config.yaml"
    weights_path = "411Fmodel.pth"  # Update this to the correct weights path

    # Initialize the tester
    tester = HRNetTester(config_path, batch_size=1, device="cuda")

    # Load weights
    tester.load_weights(weights_path)

    # Test and visualize
    tester.test_and_visualize(frame_index=100)
