import os
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset  # Ensure the correct import path

def count_total_frames(dataset):
    """
    Function to count the total number of frames in the dataset.

    Args:
        dataset (SwimmerDataset): The dataset instance to count frames from.

    Returns:
        int: Total number of frames in the dataset.
    """
    return len(dataset)

def test_dataloader(batch_size=4):
    """
    Function to test the SwimmerDataset data loader.

    Args:
        batch_size (int): Number of samples per batch for testing.
    """
    try:
        # Initialize the dataset
        dataset = SwimmerDataset()

        # Count the total frames in the dataset
        total_frames = count_total_frames(dataset)
        print(f"Total frames in the dataset: {total_frames}")

        # Initialize the data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Iterate over one batch
        for batch_idx, data in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1} Loaded:")
            print("Images shape:", data['image'].shape)
            print("Keypoints shape:", data['keypoints'].shape)
            print("Visibility shape:", data['visibility'].shape)
            print("Heatmaps shape:", data['heatmaps'].shape)
            print("Metadata:", data['meta'])

            # Only display the first batch for testing purposes
            break

        print("\nData loader test successful.")

    except Exception as e:
        print("Error occurred while testing the data loader:", e)

if __name__ == "__main__":
    # Run the test
    test_dataloader()
