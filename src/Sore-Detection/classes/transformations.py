
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch

# Important that the Transform will be the same for masks and images during augmentations

class SynchronizedTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):
        # Resize using torchvision.transforms.functional
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask
    

# Create validation transformation (because we only need resizing here)
class ValidationTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask