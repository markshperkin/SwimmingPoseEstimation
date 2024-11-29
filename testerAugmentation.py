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

        Args:
            dataset (SwimmerDataset): Instance of the SwimmerDataset class.
            augmentation (Augmentation): Instance of the Augmentation class.
        """
        self.dataset = dataset
        self.augmentation = augmentation

    def visualize_keypoints(self, ax, frame, keypoints, keypoint_names, visibility, title):
        """
        Visualize keypoints on a frame, including visibility information.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis for plotting.
            frame (numpy.ndarray): Frame (image) to display.
            keypoints (numpy.ndarray): Keypoints to overlay.
            keypoint_names (list): Names of the keypoints.
            visibility (numpy.ndarray): Visibility flags for each keypoint.
            title (str): Title of the subplot.
        """
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis('off')
        for idx, (x, y) in enumerate(keypoints):
            if x >= 0 and y >= 0:  # Ensure keypoints are valid
                color = 'red' if visibility[idx] > 0 else 'gray'  # Gray for invisible keypoints
                ax.add_patch(Circle((x, y), radius=5, color=color))
                visibility_status = f" ({visibility[idx]})"  # Append visibility status
                ax.text(x, y - 10, keypoint_names[idx] + visibility_status, color='blue', fontsize=8)

    def test_augmentation(self, augmentation_function, function_name, *args):
        """
        Test and visualize a specific augmentation function.

        Args:
            augmentation_function (function): Augmentation function to test.
            function_name (str): Name of the augmentation function.
            *args: Additional arguments for the augmentation function.
        """
        # Select a random sample from the dataset
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        image = sample['image'].permute(1, 2, 0).numpy()  # Convert [C, H, W] to [H, W, C]
        keypoints = sample['keypoints'].numpy()
        visibility = sample['visibility'].numpy()

        # Define keypoint names (based on IDs in the dataset)
        keypoint_names = [
            "Lhand", "Rhand", "Lelbow", "Relbow", "Lshoulder", "Rshoulder",
            "head", "Lhip", "Rhip", "Lknee", "Rknee", "Lancle", "Rancle"
        ]

        # Visualize before augmentation
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        self.visualize_keypoints(axes[0], image, keypoints, keypoint_names, visibility, f"Before {function_name}")

        # Apply the augmentation function
        augmented_image, augmented_keypoints, augmented_visibility = augmentation_function(image, keypoints, visibility, *args)

        # Visualize after augmentation
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


# Ensure the new augmentation methods exist

def rotate_image_and_keypoints(image, keypoints, visibility):
    return Augmentation().rotate_image_and_keypoints(image, keypoints, visibility, angle=random.uniform(-30, 30))

def translate_image_and_keypoints(image, keypoints, visibility):
    return Augmentation().translate_image_and_keypoints(image, keypoints, visibility, tx=random.randint(-10, 10), ty=random.randint(-10, 10))


if __name__ == "__main__":
    # Initialize the dataset
    dataset = SwimmerDataset()

    # Initialize augmentation
    augmentation = Augmentation()

    # Run the TesterAugmentation
    tester = TesterAugmentation(dataset, augmentation)
    tester.run()
