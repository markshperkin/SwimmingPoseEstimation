# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
import os
from torch.utils.data import DataLoader
from dataloader import SwimmerDataset
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def count_total_frames(dataset):
    """
    Function to count the total number of frames in the dataset.
    """
    return len(dataset)

def visualize_frame_with_annotations(image, keypoints, visibility, title="Frame with Annotations"):
    """
    Visualizes a frame with keypoints and their visibility.
    """
    # convert image tensor to NumPy array for visualization [C, H, W] → [H, W, C]
    image_np = image.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis('off')
    
    # overlay keypoints
    for idx, (x, y) in enumerate(keypoints):
        if visibility[idx] > 0:  # only show visible keypoints
            color = "red" if visibility[idx] == 2 else "blue"  # red for fully visible, blue for partially visible
            ax.add_patch(Circle((x, y), radius=3, color=color))
            ax.text(x, y - 5, f"{idx}", color="yellow", fontsize=8)

    plt.show()

def test_dataloader(batch_size=4):
    """
    Function to test the SwimmerDataset data loader.
    """
    try:
        # initialize the dataset
        dataset = SwimmerDataset()

        # count the total frames in the dataset
        total_frames = count_total_frames(dataset)
        print(f"Total frames in the dataset inside tester: {total_frames}")

        # initialize the data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # iterate over one batch
        for batch_idx, data in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1} Loaded:")

            for frame_idx, (image, keypoints, visibility, meta) in enumerate(
                zip(data['image'], data['keypoints'], data['visibility'], data['meta']['image_id'])
            ):
                # print frame index and metadata
                print(f"Frame Index: {frame_idx}, Frame ID: {meta}")
                
                # check if frame index exceeds 388 (dataset doubled)
                if frame_idx >= 388:
                    print(f"Frame Index {frame_idx} indicates flipped data.")

                # visualize the frame with annotations
                visualize_frame_with_annotations(image, keypoints.numpy(), visibility.numpy(), title=f"Frame {meta}")

            # only display the first batch for testing purposes
            break

        print("\nData loader test successful.")

    except Exception as e:
        print("Error occurred while testing the data loader:", e)

if __name__ == "__main__":
    # run the test
    test_dataloader()
