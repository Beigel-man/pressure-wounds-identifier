from skimage.measure import label, regionprops
import numpy as np
import cv2
import torch
from scipy import ndimage

def post_process_mask(mask):
    """
    Post-process the predicted mask with improved shape handling
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Initial thresholding
    thresh_val = 0.2

    mask_binary = (mask < thresh_val).astype(np.uint8)
    # Find initial components
    labeled_mask = label(mask_binary)
    props = regionprops(labeled_mask)
    
    if not props:
        return np.zeros_like(mask_binary)
    
    # Filter components by size
    image_area = mask.shape[0] * mask.shape[1]
    min_size = image_area * 0.005  # Min 0.1% of image
    max_size = image_area * 0.2    # Max 10% of image
    
    # Get valid components and their centroids
    valid_components = []
    centroids = []
    for prop in props:
        if min_size <= prop.area <= max_size:
            valid_components.append(prop)
            centroids.append(prop.centroid)
    
    if not valid_components:
        return np.zeros_like(mask_binary)
    
    # Create initial mask with valid components
    final_mask = np.zeros_like(mask_binary)
    
    # If we have multiple components, check if they should be merged
    if len(valid_components) > 1:
        # Calculate distances between centroids
        from scipy.spatial.distance import pdist
        distances = pdist([list(c) for c in centroids])
        
        # If components are close (within 50 pixels), merge them with morphological operations
        if np.min(distances) < 50:
            for prop in valid_components:
                final_mask[labeled_mask == prop.label] = 1
            # Apply closing to merge nearby components
            kernel_size = int(np.min(distances) // 2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        else:
            # If components are far apart, just keep the largest one
            largest = max(valid_components, key=lambda x: x.area)
            final_mask[labeled_mask == largest.label] = 1
    else:
        # If single component, just use it
        final_mask[labeled_mask == valid_components[0].label] = 1
    
    # Fill holes
    final_mask = ndimage.binary_fill_holes(final_mask)
    
    final_mask = 1- final_mask

    return final_mask.astype(np.uint8)