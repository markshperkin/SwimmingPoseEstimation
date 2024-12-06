# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataloader import SwimmerDataset
from Augmentation import Augmentation


class TesterAugmentation:
    """
    Class to test the Augmentation class by loading data from the SwimmerDataset,
    applying various augmentations, and visualizing the results.
    """
    def __init__(self, dataset, augmentation):
        """
        Initialize TesterAugmentation with a dataset and augmentation instance.
        """
        self.dataset = dataset
        self.augmentation = augmentation

    def visualize_keypoints(self, ax, frame, keypoints, keypoint_names, visibility, title):
        """
        Visualize keypoints on a frame, including visibility information.
        """
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis('off')
        for idx, (x, y) in enumerate(keypoints):
            if x >= 0 and y >= 0:
                color = 'red' if visibility[idx] > 0 else 'gray'  # gray for invisible keypoints
                ax.add_patch(Circle((x, y), radius=5, color=color))
                visibility_status = f" ({visibility[idx]})"
                ax.text(x, y - 10, keypoint_names[idx] + visibility_status, color='blue', fontsize=8)

    def test_augmentation(self, augmentation_function, function_name, *args):
        """
        Test and visualize a specific augmentation function.
        """
        # select a random sample from the dataset
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        image = sample['image'].permute(1, 2, 0).numpy()  # convert [C, H, W] to [H, W, C]
        keypoints = sample['keypoints'].numpy()
        visibility = sample['visibility'].numpy()

        # define keypoint names
        keypoint_names = [
            "Lhand", "Rhand", "Lelbow", "Relbow", "Lshoulder", "Rshoulder",
            "head", "Lhip", "Rhip", "Lknee", "Rknee", "Lancle", "Rancle"
        ]

        # visualize before augmentation
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        self.visualize_keypoints(axes[0], image, keypoints, keypoint_names, visibility, f"Before {function_name}")

        # apply the augmentation function
        augmented_image, augmented_keypoints, augmented_visibility = augmentation_function(image, keypoints, visibility, *args)

        # visualize after augmentation
        self.visualize_keypoints(axes[1], augmented_image, augmented_keypoints, keypoint_names, augmented_visibility, f"After {function_name}")
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run the tests for all augmentations.
        """
        print("Testing Horizontal Flip...")
        self.test_augmentation(self.augmentation.horizontalFlip, "Horizontal Flip")

        print("Testing Rotation...")
        self.test_augmentation(self.augmentation.rotate_image_and_keypoints, "Rotation")

        print("Testing Translation...")
        self.test_augmentation(self.augmentation.translate_image_and_keypoints, "Translation")


if __name__ == "__main__":
    # initialize the dataset
    dataset = SwimmerDataset()

    # initialize augmentation
    augmentation = Augmentation()

    # run the TesterAugmentation
    tester = TesterAugmentation(dataset, augmentation)
    tester.run()
