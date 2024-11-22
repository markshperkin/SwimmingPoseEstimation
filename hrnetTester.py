import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset  # Ensure this is the correct path
from pose_hrnet import get_pose_net  # Ensure this is the correct path to HRNet implementation


class HRNetTester:
    def __init__(self, config_path, dataset_path, batch_size=1, device="cuda"):
        """
        Initializes the HRNetTester class.

        Args:
            config_path (str): Path to the configuration YAML file.
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for the dataloader.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # Load the configuration file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize the dataset and dataloader
        self.dataset = SwimmerDataset(dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Load the HRNet model
        self.model = get_pose_net(self.config)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def test_and_visualize(self):
        """
        Tests the HRNet model on a single batch and visualizes the results.
        """
        # Load one sample batch
        sample = next(iter(self.dataloader))
        image = sample["image"].to(self.device)  # [B, C, H, W]
        ground_truth_heatmaps = sample["heatmaps"]  # [B, K, H, W]
        meta = sample["meta"]  # Metadata for debugging purposes

        # Forward pass through the model
        with torch.no_grad():
            predicted_heatmaps = self.model(image)  # Output shape: [B, K, H, W]

        # Visualize the results
        batch_idx = 0  # Display the first image in the batch
        num_keypoints = predicted_heatmaps.shape[1]

        fig, axs = plt.subplots(4, num_keypoints + 1, figsize=(30, 15))
        fig.suptitle("Original Image, Keypoints, Ground Truth Heatmaps, and Predicted Heatmaps", fontsize=16)

        # Display the original image
        axs[0, 0].imshow(image[batch_idx].permute(1, 2, 0).cpu().numpy())
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        for i in range(num_keypoints):
            # Ground Truth Heatmap
            axs[1, i + 1].imshow(ground_truth_heatmaps[batch_idx, i].cpu().numpy(), cmap="hot")
            axs[1, i + 1].set_title(f"GT Keypoint {i + 1}")
            axs[1, i + 1].axis("off")

            # Predicted Heatmap
            axs[2, i + 1].imshow(predicted_heatmaps[batch_idx, i].cpu().numpy(), cmap="hot")
            axs[2, i + 1].set_title(f"Pred Keypoint {i + 1}")
            axs[2, i + 1].axis("off")

        # Overlay keypoints on the original image
        original_image = image[batch_idx].permute(1, 2, 0).cpu().numpy()
        axs[3, 0].imshow(original_image)
        keypoints = sample["keypoints"][batch_idx].cpu().numpy()
        for kp in keypoints:
            axs[3, 0].plot(kp[0], kp[1], "ro")  # Plot keypoints in red
        axs[3, 0].set_title("Keypoints on Image")
        axs[3, 0].axis("off")

        plt.tight_layout()
        plt.show()

        # Print metadata for debugging
        print("Metadata for the sample image:")
        print(meta[batch_idx])


if __name__ == "__main__":
    # Path to config.yaml and dataset
    config_path = "config.yaml"
    dataset_path = "annotations/annotations/person_keypoints_Train.json"

    # Ensure the configuration file exists
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

    # Write the configuration to a file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Initialize the tester
    tester = HRNetTester(config_path, dataset_path, batch_size=1, device="cuda")

    # Test and visualize
    tester.test_and_visualize()
