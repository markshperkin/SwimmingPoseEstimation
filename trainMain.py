from dataloader import SwimmerDataset
from torch.utils.data import DataLoader
from train import HRNetTrainer
from testerCNN import HRNetKeypointDetector

# Instantiate dataset and dataloader
dataset = SwimmerDataset(annotations_file='annotations/annotations/person_keypoints_Train.json',
                         image_dir='annotations/images/Train')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Instantiate model
model = HRNetKeypointDetector(num_keypoints=13)

# Instantiate trainer
trainer = HRNetTrainer(model, dataloader, device='cuda', lr=1e-4)

# Train the model
trainer.train(num_epochs=10, save_path="hrnet_model")

# Evaluate on validation/test dataset
# validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
# trainer.evaluate(validation_dataloader)
