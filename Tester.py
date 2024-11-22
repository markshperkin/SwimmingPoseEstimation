import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from pose_hrnet import get_pose_net  # Ensure this imports the HRNet implementation
import yaml
from PIL import Image
import os

class VideoPoseEstimator:
    def __init__(self, model, config, device="cuda", image_size=(256, 256)):
        """
        Initialize the video pose estimator.

        Args:
            model (torch.nn.Module): Trained HRNet model.
            config (dict): Configuration dictionary for the model.
            device (str): Device to run the model on ('cuda' or 'cpu').
            image_size (tuple): Target size (height, width) for resizing frames.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.model = model.to(self.device)
        self.model.eval()

        # Preprocessing transforms
        self.transforms = Compose([
            Resize(self.image_size),
            ToTensor()
        ])

    def process_frame(self, frame):
        """
        Process a single frame for pose estimation.

        Args:
            frame (numpy.ndarray): The input video frame (H, W, C).

        Returns:
            numpy.ndarray: Frame with pose estimations overlaid.
        """
        original_frame = frame.copy()
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = self.transforms(frame).unsqueeze(0).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            predicted_heatmaps = self.model(frame_tensor)

        # Process heatmaps to extract keypoints
        keypoints = self.extract_keypoints(predicted_heatmaps[0])

        # Define skeleton connections based on annotation
        skeleton = [[9, 7], [2, 0], [11, 9], [12, 10], [10, 8], [5, 3],
                    [3, 1], [4, 6], [6, 5], [4, 5], [8, 7], [4, 2],
                    [5, 8], [7, 4]]


        # Overlay keypoints and skeleton on the frame
        output_frame = self.overlay_keypoints(original_frame, keypoints, heatmap_size=(64, 64))
        output_frame = self.overlay_skeleton(output_frame, keypoints, skeleton, heatmap_size=(64, 64))

        return output_frame



    def extract_keypoints(self, heatmaps):
        """
        Extract keypoints from heatmaps.

        Args:
            heatmaps (torch.Tensor): Predicted heatmaps of shape [K, H, W].

        Returns:
            list: List of keypoint coordinates [(x1, y1), (x2, y2), ...].
        """
        keypoints = []
        for i in range(heatmaps.shape[0]):  # Iterate over each keypoint
            heatmap = heatmaps[i].cpu().numpy()
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            keypoints.append((x, y))  # Append (x, y) coordinates
        return keypoints

    def overlay_keypoints(self, frame, keypoints, heatmap_size=(64, 64)):
        """
        Overlay keypoints and their names on the frame with colors.

        Args:
            frame (numpy.ndarray): Original video frame.
            keypoints (list): List of keypoint coordinates [(x1, y1), ...].
            heatmap_size (tuple): Size of the predicted heatmaps (H, W).

        Returns:
            numpy.ndarray: Frame with keypoints and names overlaid.
        """
        frame_height, frame_width = frame.shape[:2]
        heatmap_width, heatmap_height = heatmap_size

        # Keypoint names and corresponding colors
        keypoint_data = {
            "Lhand": (255, 0, 255),    # Purple
            "Rhand": (0, 255, 0),      # Green
            "Lelbow": (255, 0, 255),   # Purple
            "Relbow": (0, 255, 0),     # Green
            "Lshoulder": (255, 0, 255), # Purple
            "Rshoulder": (0, 255, 0),  # Green
            "head": (0, 0, 255),       # Red
            "Lhip": (255, 0, 255),     # Purple
            "Rhip": (0, 255, 0),       # Green
            "Lknee": (255, 0, 255),    # Purple
            "Rknee": (0, 255, 0),      # Green
            "Lancle": (255, 0, 255),   # Purple
            "Rancle": (0, 255, 0)      # Green
        }

        keypoint_names = list(keypoint_data.keys())
        keypoint_colors = list(keypoint_data.values())

        for i, (x, y) in enumerate(keypoints):
            # Scale keypoints from heatmap size to frame size
            scaled_x = int(x * frame_width / heatmap_width)
            scaled_y = int(y * frame_height / heatmap_height)

            # Draw the keypoint as a circle
            color = keypoint_colors[i] if i < len(keypoint_colors) else (255, 255, 255)  # Default white if out of range
            cv2.circle(frame, (scaled_x, scaled_y), radius=5, color=color, thickness=-1)

            # Overlay the keypoint name
            keypoint_name = keypoint_names[i] if i < len(keypoint_names) else f"Keypoint {i+1}"
            cv2.putText(
                frame, keypoint_name, (scaled_x, scaled_y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=color, thickness=1, lineType=cv2.LINE_AA
            )

        return frame



    def overlay_skeleton(self, frame, keypoints, skeleton, heatmap_size=(64, 64)):
        """
        Overlay skeleton lines on the frame.

        Args:
            frame (numpy.ndarray): Original video frame.
            keypoints (list): List of keypoint coordinates [(x1, y1), ...].
            skeleton (list): List of connections between keypoints [(start, end), ...].
            heatmap_size (tuple): Size of the predicted heatmaps (H, W).

        Returns:
            numpy.ndarray: Frame with skeleton lines overlaid.
        """
        frame_height, frame_width = frame.shape[:2]
        heatmap_width, heatmap_height = heatmap_size

        for start_idx, end_idx in skeleton:
            if keypoints[start_idx][0] is None or keypoints[end_idx][0] is None:
                continue  # Skip missing keypoints

            start_x = int(keypoints[start_idx][0] * frame_width / heatmap_width)
            start_y = int(keypoints[start_idx][1] * frame_height / heatmap_height)
            end_x = int(keypoints[end_idx][0] * frame_width / heatmap_width)
            end_y = int(keypoints[end_idx][1] * frame_height / heatmap_height)

            cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(255, 255, 255), thickness=2)

        return frame


    def process_video(self, input_video_path, output_video_path):
        """
        Process a video and save the output with pose estimations.

        Args:
            input_video_path (str): Path to the input video file.
            output_video_path (str): Path to save the output video file.
        """
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load the trained model
    model = get_pose_net(config, is_train=False)
    model.load_state_dict(torch.load("hrnet_checkpoint.pth"))  # Replace with your model's checkpoint path

    # Initialize the video pose estimator
    video_pose_estimator = VideoPoseEstimator(model, config)

    # Input and output video paths
    input_video_path = "13_2_10_17.mov"  # Replace with your input video path
    output_video_path = "output_pose_video2.mp4"  # Replace with your desired output video path

    # Process the video
    video_pose_estimator.process_video(input_video_path, output_video_path)
