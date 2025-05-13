import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from classes.transformations import SynchronizedTransform, ValidationTransform
from functions.nn import get_pretrained_unet, combined_loss
from functions.nn import train_model
from functions.nn import visualize_loss_curves
from functions.nn import check_validation
from classes.data import WoundDataset


# Define paths for labeled images and their masks (labels)
py_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
train_images_dir = py_folder_path + '/data/labeled/images'
train_masks_dir = py_folder_path + '/data/labeled/masks'


# Split train and validation datasets
train_files, val_files = train_test_split(os.listdir(train_images_dir), test_size=0.2, random_state=111)

# Transformations
synchronized_transform = SynchronizedTransform(size=(256, 256))
validation_transform = ValidationTransform(size=(256, 256))

# Datasets
train_dataset = WoundDataset(train_images_dir, train_masks_dir, train_files, transform=synchronized_transform)
val_dataset = WoundDataset(train_images_dir, train_masks_dir, val_files, transform=validation_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False)

# Initialize model

model = get_pretrained_unet()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Optimizer (can be one of many)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def main():
    # check if the model is on the right device
    print(f"Model is on device: {next(model.parameters()).device}")

    # Train the model
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        combined_loss=combined_loss,
        device=device,
        num_epochs=400
    )
    visualize_loss_curves(train_losses, val_losses)

    # Check on validation set
    check_validation(trained_model, val_loader, device)

    # Save the trained model
    torch.save(trained_model.state_dict(), py_folder_path + '/results/trained_model.pth')

if __name__ == '__main__':
    main()


# Leave the GPU memory free
torch.cuda.empty_cache()
torch.cuda.synchronize()


