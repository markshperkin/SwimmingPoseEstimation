# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
import cv2
import numpy as np

class Augmentation:
    """
    Augmentation class for handling image and keypoint augmentations.
    """
    def __init__(self):
        """
        Initialize the Augmentation class with matched parts for left-right swaps.
        """
        self.matched_parts = [
            (0, 1),  # Lhand <-> Rhand
            (2, 3),  # Lelbow <-> Relbow
            (4, 5),  # Lshoulder <-> Rshoulder
            (7, 8),  # Lhip <-> Rhip
            (9, 10), # Lknee <-> Rknee
            (11, 12) # Lancle <-> Rancle
        ]

    def horizontalFlip(self, frame, keypoints, visibility):
        """
        Flip the image and keypoints horizontally while swapping left-right keypoint IDs.
        """
        flipped_frame = cv2.flip(frame, 1)
        width = frame.shape[1]

        flipped_keypoints = keypoints.copy()
        flipped_keypoints[:, 0] = width - keypoints[:, 0] - 1

        for left_id, right_id in self.matched_parts:
            flipped_keypoints[left_id, :], flipped_keypoints[right_id, :] = (
                flipped_keypoints[right_id, :].copy(),
                flipped_keypoints[left_id, :].copy(),
            )
            visibility[left_id], visibility[right_id] = visibility[right_id], visibility[left_id]

        return flipped_frame, flipped_keypoints, visibility


    def rotate_image_and_keypoints(self, image, keypoints, visibility, angle=None):
        """
        Rotate the image and adjust keypoints accordingly.
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        if angle is None:
            angle = np.random.uniform(-30, 30)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        ones = np.ones((keypoints.shape[0], 1))
        keypoints_homogeneous = np.hstack([keypoints, ones])
        rotated_keypoints = np.dot(rotation_matrix, keypoints_homogeneous.T).T

        return rotated_image, rotated_keypoints, visibility

    def translate_image_and_keypoints(self, image, keypoints, visibility, tx=None, ty=None):
        """
        Translate the image and adjust keypoints accordingly.
        """
        h, w = image.shape[:2]
        if tx is None:
            tx = np.random.randint(-int(0.1 * w), int(0.1 * w))
        if ty is None:
            ty = np.random.randint(-int(0.1 * h), int(0.1 * h))

        translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        translated_keypoints = keypoints + np.array([tx, ty])

        return translated_image, translated_keypoints, visibility
