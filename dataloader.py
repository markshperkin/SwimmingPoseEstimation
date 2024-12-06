# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch
from Augmentation import Augmentation

class SwimmerDataset(Dataset):
    """
    Dataset class for loading and processing the COCO-format swimmer dataset
    with 13 keypoints and integrated data augmentation.
    """
    def __init__(self, root_dir='data', image_size=(256, 256), augmentation=True):
        """
        Initialize the dataset.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augmentation = augmentation
        self.video_dirs = self._discover_video_directories()
        self.annotations = self._load_annotations()
        self.image_ids = list(self.annotations.keys())
        self.augmenter = Augmentation()

        # double the dataset by flipping and appending flipped data
        self._double_dataset_with_flip()



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
        total_frames = 0

        for video_dir in self.video_dirs:
            annotations_dir = os.path.join(video_dir, 'annotations')
            images_dir = os.path.join(video_dir, 'images')

            annotation_files = [
                f for f in os.listdir(annotations_dir) if f.endswith('.json')
            ]
            if len(annotation_files) != 1:
                print(f"Skipping {video_dir}: Expected 1 annotation file, found {len(annotation_files)}.")
                continue

            annotation_path = os.path.join(annotations_dir, annotation_files[0])

            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)

            for image in annotation_data['images']:
                image_id = f"{os.path.basename(video_dir)}_{image['id']}"
                img_annotations = [
                    ann for ann in annotation_data['annotations'] if ann['image_id'] == image['id']
                ]
                annotations[image_id] = {
                    'image_info': image,
                    'annotations': img_annotations,
                    'image_dir': images_dir,
                    'annotation_file': annotation_path
                }
                if len(img_annotations) == 0:
                    print(f"Warning: No annotations found for image ID {image_id}.")
                else:
                    total_frames += 1

        print(f"Total number of frames successfully loaded: {total_frames}")
        return annotations

    def _double_dataset_with_flip(self):
        """
        Double the dataset by flipping all frames and appending them as new samples.
        """
        flipped_annotations = {}
        for image_id, data in self.annotations.items():
            # load original image and keypoints
            image_path = os.path.join(data['image_dir'], data['image_info']['file_name'])
            image = self.load_image(image_path)
            keypoints, visibility = self.process_keypoints(data['annotations'], data['image_info'])

            # apply horizontal flip
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
            flipped_image_np, flipped_keypoints, flipped_visibility = self.augmenter.horizontalFlip(image_np, keypoints, visibility)

            # prepare flipped metadata
            flipped_image_id = f"FLIPPED_{image_id}"
            flipped_annotations[flipped_image_id] = {
                'image_info': {
                    'file_name': f"FLIPPED_{data['image_info']['file_name']}",
                    'width': data['image_info']['width'],
                    'height': data['image_info']['height']
                },
                'annotations': [{
                    'keypoints': flipped_keypoints.flatten().tolist(),
                    'visibility': flipped_visibility.tolist()
                }],
                'image_dir': data['image_dir'],
                'annotation_file': data['annotation_file']
            }

        # append flipped data to the original annotations
        self.annotations.update(flipped_annotations)
        self.image_ids.extend(flipped_annotations.keys())
        print(f"Dataset doubled with flipped data. Total frames: {len(self.image_ids)}")

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding keypoints and annotations.
        Returns:
            dict: Processed image, keypoints, visibility, and metadata.
        """
        # check if the sample is flipped based on the image ID
        is_flipped_sample = self.image_ids[idx].startswith("FLIPPED_")
        original_image_id = self.image_ids[idx].replace("FLIPPED_", "") if is_flipped_sample else self.image_ids[idx]
        
        annotation_data = self.annotations[original_image_id]
        image_info = annotation_data['image_info']
        annotations = annotation_data['annotations']
        image_dir = annotation_data['image_dir']

        # load and resize the image
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = self.load_image(image_path)

        # load and process keypoints
        keypoints, visibility = self.process_keypoints(annotations, image_info)

        # apply flipping dynamically if this is a flipped sample
        if is_flipped_sample:
            # convert tensor to NumPy for augmentation
            image_np = image.permute(1, 2, 0).numpy() * 255  # [C, H, W] → [H, W, C], scale [0, 1] → [0, 255]
            image_np = image_np.astype(np.uint8)
            image_np, keypoints, visibility = self.augmenter.horizontalFlip(image_np, keypoints, visibility)
            image = torch.tensor(image_np / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # apply rotation and translation augmentations
        if np.random.rand() < 0.1:  # 10% chance for translation
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
            image_np, keypoints, visibility = self.augmenter.translate_image_and_keypoints(image_np, keypoints, visibility)
            image = torch.tensor(image_np / 255.0, dtype=torch.float32).permute(2, 0, 1)


        if np.random.rand() < 0.2:  # 20% chance for rotation
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
            image_np, keypoints, visibility = self.augmenter.rotate_image_and_keypoints(image_np, keypoints, visibility)
            image = torch.tensor(image_np / 255.0, dtype=torch.float32).permute(2, 0, 1)


        # create heatmaps for keypoints
        heatmaps = self.create_heatmaps(keypoints, visibility)

        # downsample heatmaps to 64 x 64
        heatmaps_downsampled = F.interpolate(
            torch.tensor(heatmaps).unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).numpy()

        return {
            'image': image, 
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'visibility': torch.tensor(visibility, dtype=torch.int32),
            'heatmaps': torch.tensor(heatmaps_downsampled, dtype=torch.float32),
            'meta': {
                'image_id': self.image_ids[idx],
                'original_size': (image_info['width'], image_info['height']),
            }
        }

    def load_image(self, image_path):
        """
        Load an image from a given path, resize to a consistent resolution, and convert to PyTorch tensor format [C, H, W].
        """
        transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor() # Convert image to [C, H, W] and normalize to [0, 1]
        ])
        return transform(Image.open(image_path).convert('RGB'))

    def process_keypoints(self, annotations, image_info):
        """
        Extract keypoints and their visibility from annotations.
        """
        if not annotations:
            # if no annotations, return empty keypoints and visibility
            print(f"Warning: No annotations found for image ID {image_info['id']}.")
            num_keypoints = 13
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


        # scale keypoints to match resized image dimensions
        orig_width, orig_height = image_info['width'], image_info['height']
        scale_x, scale_y = self.image_size[1] / orig_width, self.image_size[0] / orig_height
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return keypoints, visibility


    def create_heatmaps(self, keypoints, visibility):
        """
        Generate heatmaps for each keypoint, resized to match the image dimensions.
        """
        height, width = self.image_size
        heatmaps = np.zeros((len(keypoints), height, width), dtype=np.float32)

        for i, (kp, vis) in enumerate(zip(keypoints, visibility)):
            if vis > 0:  # only create heatmap for visible keypoints
                heatmaps[i] = self.gaussian_heatmap(kp, self.image_size)

        return heatmaps

    def gaussian_heatmap(self, keypoint, image_size, sigma=2):
        """
        Generate a Gaussian heatmap for a single keypoint.
        """
        height, width = image_size
        x, y = int(keypoint[0]), int(keypoint[1])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return heatmap
