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
    def __init__(self, root_dir='data', image_size=(256, 256)):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory containing video directories.
            image_size (tuple): Target size (height, width) for resizing images and heatmaps.
        """
        self.root_dir = root_dir
        self.image_size = image_size  # Target image size (height, width)
        self.video_dirs = self._discover_video_directories()
        self.annotations = self._load_annotations()
        self.image_ids = list(self.annotations.keys())

    def _discover_video_directories(self):
        """
        Discover all video directories in the root directory.

        Returns:
            list: List of video directory paths.
        """
        video_dirs = []
        for entry in os.scandir(self.root_dir):
            if entry.is_dir():
                video_dirs.append(entry.path)
        return video_dirs

    def _load_annotations(self):
        """
        Load annotations and associate them with corresponding frames.

        Returns:
            dict: Dictionary mapping image IDs to annotation data.
        """
        annotations = {}
        total_frames = 0  # Counter for the total number of frames loaded

        for video_dir in self.video_dirs:
            annotations_dir = os.path.join(video_dir, 'annotations')
            images_dir = os.path.join(video_dir, 'images')

            # Find the annotation JSON file (there is only one per video directory)
            annotation_files = [
                f for f in os.listdir(annotations_dir) if f.endswith('.json')
            ]
            if len(annotation_files) != 1:
                print(f"Skipping {video_dir}: Expected 1 annotation file, found {len(annotation_files)}.")
                continue  # Skip if no valid annotation file

            annotation_path = os.path.join(annotations_dir, annotation_files[0])

            # Load the annotation JSON file
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)

            # Map image IDs to annotations
            for image in annotation_data['images']:
                # Add video-specific prefix to image_id
                image_id = f"{os.path.basename(video_dir)}_{image['id']}"  
                img_annotations = [
                    ann for ann in annotation_data['annotations'] if ann['image_id'] == image['id']
                ]
                
                # Only create entry if annotations exist
                annotations[image_id] = {
                    'image_info': image,
                    'annotations': img_annotations,
                    'image_dir': images_dir,
                    'annotation_file': annotation_path
                }

                # Print debug information
                print(f"Loaded annotations for image ID {image_id} from {annotation_path}.")
                if len(img_annotations) == 0:
                    print(f"Warning: No annotations found for image ID {image_id}.")
                else:
                    total_frames += 1

        # Log the total number of loaded frames
        print(f"Total number of frames successfully loaded: {total_frames}")
        print(f"Unique image IDs in the dataset: {len(annotations.keys())}")
        return annotations


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
        annotation_data = self.annotations[image_id]
        image_info = annotation_data['image_info']
        annotations = annotation_data['annotations']
        image_dir = annotation_data['image_dir']
        annotation_file = annotation_data['annotation_file']

        # Load and resize image
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = self.load_image(image_path)

        # Load and process keypoints
        keypoints, visibility = self.process_keypoints(annotations, image_info)

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
                'image_and_annotation': f"Image: {image_path}, Annotation File: {annotation_file}",  # Image and annotation on the same line
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

    def process_keypoints(self, annotations, image_info):
        """
        Extract keypoints and their visibility from annotations.

        Args:
            annotations (list): List of annotations for the image.
            image_info (dict): Information about the image (width, height).

        Returns:
            np.ndarray: Keypoints (num_keypoints x 2).
            np.ndarray: Visibility (num_keypoints).
        """
        if not annotations:
            # No annotations for this image, return empty keypoints and visibility
            print(f"Warning: No annotations found for image ID {image_info['id']}.")
            num_keypoints = 13  # Assuming 13 keypoints as per COCO format
            return np.zeros((num_keypoints, 2), dtype=np.float32), np.zeros((num_keypoints,), dtype=np.int32)

        num_keypoints = len(annotations[0]['keypoints']) // 3
        keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        visibility = np.zeros((num_keypoints,), dtype=np.int32)

        for ann in annotations:
            if 'keypoints' not in ann or len(ann['keypoints']) < num_keypoints * 3:
                print(f"Warning: Invalid or missing keypoints in annotation for image ID {image_info['id']}.")
                continue

            kp = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints[:, :2] = kp[:, :2]
            visibility[:] = kp[:, 2]

        # Scale keypoints to match resized image dimensions
        orig_width, orig_height = image_info['width'], image_info['height']
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
        heatmaps = np.zeros((len(keypoints), height, width), dtype=np.float32)

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
