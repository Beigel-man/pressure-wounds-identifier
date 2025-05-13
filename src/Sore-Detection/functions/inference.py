import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from pathlib import Path
from PIL import Image


def inference(model, input_dir, output_dir, device, post_process_mask=lambda x: x):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Process each image
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, img_name)
            image = Image.open(img_path).convert('RGB')

            # Ensure no flipping transformations during inference
            image_tensor = TF.resize(image, (256, 256), interpolation=InterpolationMode.BILINEAR)
            image_tensor = TF.to_tensor(image_tensor).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(image_tensor)
                output = torch.sigmoid(output)
            
            # Apply post-processing
            predicted_mask = post_process_mask(output[0][0])
            
            # Get original image size
            original_width, original_height = image.size
            
            # Convert predicted_mask to NumPy array if it's a PyTorch tensor
            if isinstance(predicted_mask, torch.Tensor):
                predicted_mask = predicted_mask.cpu().numpy()

            # Resize the predicted mask to the original image size using nearest-neighbor interpolation
            predicted_mask_resized = cv2.resize(predicted_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
            # Save the original image
            original_image_path = os.path.join(output_dir, f'{Path(img_name).stem}_original.jpg')
            image.save(original_image_path)
            
            # Save the predicted mask
            mask_path = os.path.join(output_dir, f'{Path(img_name).stem}_MASK.jpg')
            cv2.imwrite(mask_path, (predicted_mask_resized * 255).astype('uint8'))
            
            # Create overlay
            overlay = image.copy()
            predicted_mask_overlay = (predicted_mask_resized * 255).astype('uint8')
            predicted_mask_overlay = cv2.cvtColor(predicted_mask_overlay, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(np.array(overlay), 0.7, predicted_mask_overlay, 0.3, 0)

            # Save the overlay
            overlay_path = os.path.join(output_dir, f'{Path(img_name).stem}_overlay.jpg')
            cv2.imwrite(overlay_path, overlay)

            print(f'Processed {img_name}, saved original image to {original_image_path} and mask to {mask_path}')