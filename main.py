from torch.utils.data import DataLoader
from dataloader import SwimmerDataset
from testerCNN import HRNetKeypointDetector, test_pipeline_with_model
import torch

def main():
    """
    Entry point for testing the HRNet-based keypoint detection pipeline.
    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the dataset and dataloader
    dataset = SwimmerDataset(annotations_file='annotations/annotations/person_keypoints_Train.json', 
                             image_dir='annotations/images/Train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the model
    model = HRNetKeypointDetector(num_keypoints=13)

    # Test the pipeline
    test_pipeline_with_model(dataloader, model, device=device)

if __name__ == "__main__":
    main()
