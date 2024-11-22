import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np




class HRNetKeypointDetector(nn.Module):
    def __init__(self, num_keypoints):
        super(HRNetKeypointDetector, self).__init__()
        
        # Initial stem layers (strided convolutions)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Stage 1: High-resolution subnet (maintain 64 channels initially)
        self.stage1 = self._make_stage(64, 64)

        # Stage 2: Add low-resolution subnet and connect
        self.stage2 = nn.ModuleList([
            self._make_stage(64, 32),  # High-resolution
            self._make_stage(64, 64)  # Low-resolution
        ])
        
        # Fusion block for Stage 2
        self.fusion2 = self._make_fusion([32, 64], [32])
        
        # Stage 3: Add another resolution and connect
        self.stage3 = nn.ModuleList([
            self._make_stage(32, 32),  # High-resolution
            self._make_stage(64, 64),  # Medium-resolution
            self._make_stage(64, 128)  # Low-resolution
        ])
        
        # Fusion block for Stage 3
        self.fusion3 = self._make_fusion([32, 64, 128], [32])
        
        # Final regression layer
        self.regressor = nn.Sequential(
            nn.Conv2d(32, num_keypoints, kernel_size=1, stride=1),
            nn.Sigmoid()  # Heatmaps in the range [0, 1]
        )

    def _make_stage(self, in_channels, out_channels, num_blocks=4):
        """
        Create a stage consisting of residual blocks.
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(self._make_residual_block(in_channels, out_channels))
            in_channels = out_channels  # Update in_channels for subsequent blocks
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, in_channels, out_channels):
        """
        Create a basic residual block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _make_fusion(self, in_channels_list, out_channels_list):
        """
        Create a fusion layer that combines multiple resolutions.
        """
        fusion_layers = nn.ModuleList()
        for out_channels in out_channels_list:
            fusion_layers.append(nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1, stride=1))
        return fusion_layers

    def forward(self, x):
        # Stem
        x = self.stem(x)
        print("Shape after stem:", x.shape)  # Debugging output

        # Stage 1
        high_res = self.stage1(x)
        print("Shape after stage1:", high_res.shape)  # Debugging output

        # Stage 2
        high_res_2 = self.stage2[0](high_res)
        low_res_2 = self.stage2[1](F.avg_pool2d(high_res, kernel_size=2, stride=2))
        fused2 = torch.cat([high_res_2, F.interpolate(low_res_2, scale_factor=2)], dim=1)
        fused2 = self.fusion2[0](fused2)  # Ensure fused2 has 32 channels
        print("Shape after fused2:", fused2.shape)  # Debugging output

        # Stage 3
        high_res_3 = self.stage3[0](fused2)
        print("Shape after high_res_3:", high_res_3.shape)  # Debugging output

        # Transform fused2 channels to match self.stage3[1] input
        medium_res_input = F.conv2d(fused2, weight=torch.randn(64, 32, 1, 1).to(fused2.device), bias=None)
        print("Shape after transforming for medium_res_3:", medium_res_input.shape)  # Debugging output

        medium_res_3 = self.stage3[1](F.avg_pool2d(medium_res_input, kernel_size=2, stride=2))
        print("Shape after medium_res_3:", medium_res_3.shape)  # Debugging output

        low_res_3 = self.stage3[2](F.avg_pool2d(medium_res_3, kernel_size=2, stride=2))
        fused3 = torch.cat([high_res_3, F.interpolate(medium_res_3, scale_factor=2),
                            F.interpolate(low_res_3, scale_factor=4)], dim=1)
        fused3 = self.fusion3[0](fused3)
        print("Shape after fused3:", fused3.shape)  # Debugging output

        # Regress heatmaps
        heatmaps = self.regressor(fused3)
        return heatmaps



def test_pipeline_with_model(dataloader, model, device='cuda'):
    """
    Tests the pipeline by loading one sample from the SwimmerDataset dataloader,
    passing it through the HRNet model, and visualizing the results.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for SwimmerDataset.
        model (torch.nn.Module): The HRNet model.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()

    # Load one sample batch
    sample = next(iter(dataloader))
    image = sample['image'].to(device)  # [B, C, H, W]
    ground_truth_heatmaps = sample['heatmaps']  # [B, K, H, W]
    meta = sample['meta']  # Metadata for debugging purposes
    
    # Forward pass through the model
    with torch.no_grad():
        predicted_heatmaps = model(image)  # Output shape: [B, K, H, W]

    # Visualize the results
    batch_idx = 0  # Display the first image in the batch
    num_keypoints = predicted_heatmaps.shape[1]

    fig, axs = plt.subplots(4, num_keypoints + 1, figsize=(30, 15))
    fig.suptitle("Original Image, Keypoints, Ground Truth Heatmaps, and Predicted Heatmaps", fontsize=16)

    # Display the original image
    axs[0, 0].imshow(image[batch_idx].permute(1, 2, 0).cpu().numpy())
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    for i in range(num_keypoints):
        # Ground Truth Heatmap
        axs[1, i + 1].imshow(ground_truth_heatmaps[batch_idx, i].cpu().numpy(), cmap='hot')
        axs[1, i + 1].set_title(f'GT Keypoint {i+1}')
        axs[1, i + 1].axis('off')

        # Predicted Heatmap
        axs[2, i + 1].imshow(predicted_heatmaps[batch_idx, i].cpu().numpy(), cmap='hot')
        axs[2, i + 1].set_title(f'Pred Keypoint {i+1}')
        axs[2, i + 1].axis('off')

    # Overlay keypoints on the original image
    original_image = image[batch_idx].permute(1, 2, 0).cpu().numpy()
    axs[3, 0].imshow(original_image)
    keypoints = sample['keypoints'][batch_idx].cpu().numpy()
    for kp in keypoints:
        axs[3, 0].plot(kp[0], kp[1], 'ro')  # Plot keypoints in red
    axs[3, 0].set_title("Keypoints on Image")
    axs[3, 0].axis("off")

    plt.tight_layout()
    plt.show()

    # Print metadata for debugging
    print("Metadata for the sample image:")
    print(meta[batch_idx])