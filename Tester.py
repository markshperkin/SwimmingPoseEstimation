# ------------------------------------------------------------------------------
# Written by Mark Shperkin
# ------------------------------------------------------------------------------
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from pose_hrnet import get_pose_net
import yaml
from PIL import Image

class VideoPoseEstimator:
    def __init__(self, model, config, device="cuda", image_size=(256, 256)):
        """
        Initialize the video pose estimator..
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.model = model.to(self.device)
        self.model.eval()

        # preprocessing transforms
        self.transforms = Compose([
            Resize(self.image_size),
            ToTensor()
        ])

    def process_frame(self, frame, confidence_threshold=0.3):
        """
        Process a single frame for pose estimation with a confidence threshold.
        Returns:
            numpy.ndarray: Frame with pose estimations overlaid.
        """
        original_frame = frame.copy()
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = self.transforms(frame).unsqueeze(0).to(self.device)

        # forward pass through the model
        with torch.no_grad():
            predicted_heatmaps = self.model(frame_tensor)

        # process heatmaps to extract keypoints and confidences
        keypoints, confidences = self.extract_keypoints_with_confidences(predicted_heatmaps[0])

        # apply confidence threshold
        filtered_keypoints = [
            (x, y) if conf >= confidence_threshold else (None, None)
            for (x, y), conf in zip(keypoints, confidences)
        ]

        # define skeleton
        skeleton = [[9, 7], [2, 0], [11, 9], [12, 10], [10, 8], [5, 3],
                    [3, 1], [4, 6], [6, 5], [4, 5], [8, 7], [4, 2],
                    [5, 8], [7, 4]]

        # overlay keypoints and skeleton on the frame
        output_frame = self.overlay_keypoints(original_frame, filtered_keypoints, confidences, confidence_threshold)
        output_frame = self.overlay_skeleton(output_frame, filtered_keypoints, skeleton)

        return output_frame

    def extract_keypoints_with_confidences(self, heatmaps):
        """
        Extract keypoints and their confidences from heatmaps.
        Returns:
            tuple: (list of keypoint coordinates [(x1, y1), ...],
                    list of confidence scores [c1, c2, ...]).
        """
        keypoints = []
        confidences = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i].cpu().numpy()
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[y, x]
            keypoints.append((x, y))
            confidences.append(confidence)
        return keypoints, confidences

    def overlay_keypoints(self, frame, keypoints, confidences, confidence_threshold):
        """
        Overlay keypoints and their names on the frame with colors.
        """
        frame_height, frame_width = frame.shape[:2]
        heatmap_width, heatmap_height = 64, 64

        # keypoint names and corresponding colors
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

        for i, (keypoint, confidence) in enumerate(zip(keypoints, confidences)):
            if keypoint[0] is None or confidence < confidence_threshold:
                continue  # skip keypoints below the threshold

            # scale keypoints from heatmap size to frame size
            scaled_x = int(keypoint[0] * frame_width / heatmap_width)
            scaled_y = int(keypoint[1] * frame_height / heatmap_height)

            # draw the keypoint as a circle
            color = keypoint_colors[i] if i < len(keypoint_colors) else (255, 255, 0)  # default yellow if out of range
            cv2.circle(frame, (scaled_x, scaled_y), radius=5, color=color, thickness=-1)

            # overlay the keypoint name and confidence
            keypoint_name = keypoint_names[i] if i < len(keypoint_names) else f"Keypoint {i+1}"
            cv2.putText(
                frame, f"{keypoint_name} ({confidence:.2f})", (scaled_x, scaled_y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=color, thickness=1, lineType=cv2.LINE_AA
            )

        return frame


    def overlay_skeleton(self, frame, keypoints, skeleton, heatmap_size=(64, 64)):
        """
        Overlay skeleton lines on the frame.
        """
        frame_height, frame_width = frame.shape[:2]
        heatmap_width, heatmap_height = heatmap_size

        for start_idx, end_idx in skeleton:
            if keypoints[start_idx][0] is None or keypoints[end_idx][0] is None:
                continue  # skip missing keypoints

            start_x = int(keypoints[start_idx][0] * frame_width / heatmap_width)
            start_y = int(keypoints[start_idx][1] * frame_height / heatmap_height)
            end_x = int(keypoints[end_idx][0] * frame_width / heatmap_width)
            end_y = int(keypoints[end_idx][1] * frame_height / heatmap_height)

            cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(255, 255, 255), thickness=2)

        return frame


    def process_video(self, input_video_path, output_video_path):
        """
        Process a video and save the output with pose estimations.
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

            # process each frame
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    # load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load the trained model
    model = get_pose_net(config, is_train=False)
    model.load_state_dict(torch.load("411Fmodel.pth"))  # replace with the model's path

    # initialize the video pose estimator
    video_pose_estimator = VideoPoseEstimator(model, config)

    # input and output video paths
    input_video_path = "43_2_40_96.mp4"  # replace with the input video path
    output_video_path = "output_posetest.mp4"  # replace with the output video path

    # process the video
    video_pose_estimator.process_video(input_video_path, output_video_path)
