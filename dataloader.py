import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch

class SwimmerDataset(Dataset):
    """
    Dataset class for loading and processing the COCO-format swimmer dataset
    with 13 keypoints.
    """
    def __init__(self, annotations_file='annotations/annotations/person_keypoints_Train.json', 
                 image_dir='annotations/images/Train', 
                 image_size=(256, 256)):
        """
        Initialize the dataset.

        Args:
            annotations_file (str): Path to the JSON annotations file, relative to the root.
            image_dir (str): Path to the image directory, relative to the root.
            image_size (tuple): Target size (height, width) for resizing images and heatmaps.
        """
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.image_size = image_size  # Target image size (height, width)
        
        # Load annotations
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Map image IDs to annotations
        self.image_ids = [image['id'] for image in self.annotations['images']]
        self.id_to_image = {image['id']: image for image in self.annotations['images']}
        self.id_to_annotations = {ann['image_id']: [] for ann in self.annotations['annotations']}
        for ann in self.annotations['annotations']:
            self.id_to_annotations[ann['image_id']].append(ann)

        # Define keypoints
        self.keypoints = self.annotations['categories'][0]['keypoints']
        self.num_keypoints = len(self.keypoints)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding keypoints and annotations.

        Args:
            idx (int): Index of the image.
        
        Returns:
            dict: Processed image, keypoints, visibility, and metadata.
        """
        image_id = self.image_ids[idx]
        image_info = self.id_to_image[image_id]
        annotations = self.id_to_annotations[image_id]
        
        # Load and resize image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = self.load_image(image_path)

        # Load and process keypoints
        keypoints, visibility = self.process_keypoints(annotations)

        # Create heatmaps for keypoints
        heatmaps = self.create_heatmaps(keypoints, visibility)

        # Downsample heatmaps to 64 x 64
        heatmaps_downsampled = F.interpolate(
            torch.tensor(heatmaps).unsqueeze(0),  # Add batch dimension
            size=(64, 64),  # Target size
            mode='bilinear',
            align_corners=False
        ).squeeze(0).numpy()  # Remove batch dimension

        return {
            'image': image,
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'visibility': torch.tensor(visibility, dtype=torch.int32),
            'heatmaps': torch.tensor(heatmaps_downsampled, dtype=torch.float32),
            'meta': {
                'image_id': image_id,
                'image_path': image_path,
                'original_size': (image_info['width'], image_info['height']),
            }
        }

    def load_image(self, image_path):
        """
        Load an image from a given path, resize to a consistent resolution, and convert to PyTorch tensor format [C, H, W].
        """
        transform = T.Compose([
            T.Resize(self.image_size),  # Resize to target size
            T.ToTensor()               # Convert image to [C, H, W] and normalize to [0, 1]
        ])
        return transform(Image.open(image_path).convert('RGB'))

    def process_keypoints(self, annotations):
        """
        Extract keypoints and their visibility from annotations.
        
        Args:
            annotations (list): List of annotations for the image.
        
        Returns:
            np.ndarray: Keypoints (num_keypoints x 2).
            np.ndarray: Visibility (num_keypoints).
        """
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        visibility = np.zeros((self.num_keypoints,), dtype=np.int32)

        for ann in annotations:
            kp = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints[:, :2] = kp[:, :2]
            visibility[:] = kp[:, 2]

        # Retrieve dimensions from image_info
        image_id = annotations[0]['image_id']
        image_info = self.id_to_image[image_id]  # Fetch image info using image_id
        orig_width, orig_height = image_info['width'], image_info['height']

        # Scale keypoints to match resized image dimensions
        scale_x, scale_y = self.image_size[1] / orig_width, self.image_size[0] / orig_height
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return keypoints, visibility


    def create_heatmaps(self, keypoints, visibility):
        """
        Generate heatmaps for each keypoint, resized to match the image dimensions.
        
        Args:
            keypoints (np.ndarray): Keypoint coordinates.
            visibility (np.ndarray): Visibility flags.
        
        Returns:
            np.ndarray: Heatmaps for each keypoint.
        """
        height, width = self.image_size
        heatmaps = np.zeros((self.num_keypoints, height, width), dtype=np.float32)

        for i, (kp, vis) in enumerate(zip(keypoints, visibility)):
            if vis > 0:  # Only create heatmap for visible keypoints
                heatmaps[i] = self.gaussian_heatmap(kp, self.image_size)

        return heatmaps

    def gaussian_heatmap(self, keypoint, image_size, sigma=2):
        """
        Generate a Gaussian heatmap for a single keypoint.
        
        Args:
            keypoint (tuple): Keypoint (x, y).
            image_size (tuple): (height, width) of the image.
            sigma (float): Standard deviation of the Gaussian.
        
        Returns:
            np.ndarray: Gaussian heatmap.
        """
        height, width = image_size
        x, y = int(keypoint[0]), int(keypoint[1])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return heatmap
