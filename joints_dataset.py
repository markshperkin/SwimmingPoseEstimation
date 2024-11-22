import os
import copy
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transforms import get_affine_transform, affine_transform


class JointsDataset(Dataset):
    """
    Dataset for loading images and corresponding joint annotations.
    Compatible with PyTorch's DataLoader for batching and shuffling.
    """

    def __init__(self, root, image_set, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=2, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root (str): Root directory of the dataset.
            image_set (str): Subdirectory containing the dataset (e.g., "train" or "val").
            image_size (tuple): Size of the input image (width, height).
            heatmap_size (tuple): Size of the output heatmap (width, height).
            sigma (int): Standard deviation for the Gaussian heatmaps.
            transform (callable): Optional transform to be applied on images.
        """
        self.num_joints = 13  # Assuming 13 keypoints as per your dataset
        self.pixel_std = 200

        self.root = root
        self.image_set = image_set
        self.transform = transform

        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.sigma = sigma

        self.db = self._get_db()

    def _get_db(self):
        """
        Collect the dataset into a list of samples (image paths, keypoints, etc.).
        """
        db = []
        annotations_file = os.path.join(self.root, "annotations", f"{self.image_set}.json")
        images_dir = os.path.join(self.root, "images", self.image_set)

        # Parse COCO-style annotations
        import json
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        for ann in annotations["annotations"]:
            image_id = ann["image_id"]
            joints_3d = np.array(ann["keypoints"]).reshape((-1, 3))
            joints_3d_vis = (joints_3d[:, 2] > 0).astype(np.float32).reshape((-1, 1))

            image_file = os.path.join(images_dir, f"frame_{image_id:06d}.PNG")
            center = np.array(ann["bbox"][:2]) + np.array(ann["bbox"][2:]) / 2
            scale = np.array(ann["bbox"][2:]) / self.pixel_std

            db.append({
                "image": image_file,
                "joints_3d": joints_3d,
                "joints_3d_vis": joints_3d_vis,
                "center": center,
                "scale": scale
            })

        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input, target, target_weight, meta)
        """
        db_rec = copy.deepcopy(self.db[idx])

        # Load the image
        image_file = db_rec["image"]
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            raise ValueError(f"Fail to read {image_file}")

        # Transform the image
        c = db_rec["center"]
        s = db_rec["scale"]

        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        if self.transform:
            input = self.transform(input)

        # Transform joints
        joints = db_rec["joints_3d"]
        joints_vis = db_rec["joints_3d_vis"]

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            "image": image_file,
            "joints": joints,
            "joints_vis": joints_vis,
            "center": c,
            "scale": s
        }

        return input, target, target_weight, meta

    def generate_target(self, joints, joints_vis):
        """
        Generate the heatmap for the given joints.

        Args:
            joints (numpy.ndarray): Array of joint coordinates.
            joints_vis (numpy.ndarray): Array of joint visibility flags.

        Returns:
            tuple: (target, target_weight)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)
        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            if target_weight[joint_id] > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
