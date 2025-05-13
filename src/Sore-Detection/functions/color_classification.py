import numpy as np
import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path



# function to analyze the colors of the wound in the image

def analyze_wound_colors(image, mask):

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Focus only on wound area
    wound_pixels = mask == 255
    wound_region = hsv_image[wound_pixels]
    
    # Define HSV color ranges
    color_ranges = {
        'yellow': {
            'ranges': [[(6, 50, 90), (39, 120, 190)]]
        },
        'red': {
            'ranges': [
                [(0, 145, 85), (6, 185, 110)],    # First red range
                [(175, 145, 85), (180, 185, 110)] # Second red range for wrap-around
            ]
        },
        'pink': {
            'ranges': [[(0, 65, 100), (6, 145, 255)]]
        },
        'black': {
            'ranges': [[(0, 0, 0), (180, 255, 10)]]
        }
    }
    
    distributions = {}
    color_masks = {}
    total_pixels = len(wound_region)
    unclassified_mask = np.ones(total_pixels, dtype=bool)
    
    # The order of the colors is important if there are overlapping ranges. there is no overlap here.
    for color in ['black', 'red', 'pink', 'yellow']:
        info = color_ranges[color]
        color_mask = np.zeros(total_pixels, dtype=bool)
        
        for lower, upper in info['ranges']:
            lower = np.array(lower)
            upper = np.array(upper)
            in_range = np.all((wound_region >= lower) & (wound_region <= upper), axis=1)
            color_mask |= (in_range & unclassified_mask)
            
        unclassified_mask &= ~color_mask
        
        matching_pixels = np.sum(color_mask)
        percentage = (matching_pixels / total_pixels) * 100
        distributions[color] = percentage
        
        # Create full-size mask
        full_mask = np.zeros_like(mask, dtype=bool)
        full_mask[wound_pixels] = color_mask
        color_masks[color] = full_mask
        
        print(f"\n{color.capitalize()} detection:")
        print(f"Matching pixels: {matching_pixels}")
        print(f"Percentage: {percentage:.2f}%")
    
    
    return distributions, color_masks

# A function to visualize the wound colors and their distributions

def visualize_wound_colors_detailed(image, mask, distributions, color_masks):
    """
    Create detailed visualization including color-coded regions.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axs[0,0].imshow(image)
    axs[0,0].set_title('Original Image')
    axs[0,0].axis('off')
    
    # Wound Region
    masked = cv2.bitwise_and(image, image, mask=mask)
    axs[0,1].imshow(masked)
    axs[0,1].set_title('Wound Region')
    axs[0,1].axis('off')
    
    # Create color classification visualization
    h, w = mask.shape[:2]
    color_viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define colors for visualization
    viz_colors = {
        'red': [255, 0, 0],       # Red
        'pink': [255, 192, 203],  # Pink
        'yellow': [255, 255, 0],  # Yellow
        'black': [0, 0, 0]        # Black
    }
    
    wound_area = (mask == 255)
    
    for color, cmask in color_masks.items():
        valid_area = wound_area & cmask
        if color in viz_colors:
            color_viz[valid_area] = viz_colors[color]
    
    color_viz[~wound_area] = [0, 0, 0]  # Set non-wound to black

    axs[1,0].imshow(color_viz)
    axs[1,0].set_title('Color Classification (Wound Only)')
    axs[1,0].axis('off')
    
    # Color distribution bar plot
    all_colors = ['black', 'red', 'pink', 'yellow']
    percentages = [distributions.get(c, 0) for c in all_colors]
    
    bars = axs[1,1].bar(all_colors, percentages)
    axs[1,1].set_title('Color Distribution')
    axs[1,1].set_ylabel('Percentage')
    axs[1,1].set_ylim(0, max(max(percentages) * 1.1, 0.1))
    
    for bar in bars:
        height = bar.get_height()
        axs[1,1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# confidence score between 0-1 where 1 is highest confidence
def calculate_confidence(distributions, classification):

    if classification == 'mixed':
        # For mixed, look at how close top colors are
        sorted_colors = sorted(distributions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_colors) >= 2:
            top_percent = sorted_colors[0][1]
            second_percent = sorted_colors[1][1]
            # If percentages are very close, high confidence it's truly mixed
            ratio = second_percent / top_percent if top_percent > 0 else 0
            return ratio  # Will be close to 1 for truly mixed cases
        return 0.5  # Default for mixed with only one color

    else:  # one dominant color
        sorted_colors = sorted(distributions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_colors) >= 2:
            dominant_percent = sorted_colors[0][1]
            second_percent = sorted_colors[1][1]
            
            confidence_factors = [
                # Factor 1: How dominant is the main color
                dominant_percent / 100,  # Higher percentage = higher confidence
                
                # Factor 2: How much stronger is it than second color
                (dominant_percent - second_percent) / dominant_percent if dominant_percent > 0 else 0,
                
                # Factor 3: How much of the wound is classified (not unclassified)
                sum(dist for _, dist in distributions.items()) / 100
            ]
            
            # Weighted average of factors
            return sum(confidence_factors) / len(confidence_factors)
        
        return sorted_colors[0][1] / 100 if sorted_colors else 0
    
# For process_directory function to work, first- masks should be created
# second- the input_dir should contain both images and masks
# third- the images should be named as <base_name>_original.jpg and masks as <base_name>_MASK.jpg (that's how they are saved after inference) 

def process_directory(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Process each image in the directory
    for image_file in Path(input_dir).glob('*_original.jpg'):
        base_name = image_file.name.replace('_original.jpg', '')
        mask_path = image_file.parent / f"{base_name}_MASK.jpg"
        
        # Skip if mask doesn't exist
        if not mask_path.exists():
            print(f"Skipping {base_name} - no mask found")
            continue
            
        # Load image and mask
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Threshold and invert mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        
        # Analyze colors
        distributions, color_masks = analyze_wound_colors(image, mask)
        
        # Determine dominant color
        if distributions:
            # Sort colors by percentage
            sorted_colors = sorted(distributions.items(), key=lambda x: x[1], reverse=True)
            dominant_color = sorted_colors[0][0]
            dominant_percent = sorted_colors[0][1]
            
            # Check if dominant color is more than twice the second
            if len(sorted_colors) > 1:
                second_percent = sorted_colors[1][1]
                classification = dominant_color if dominant_percent > 2 * second_percent else 'mixed'
            else:
                classification = dominant_color
        else:
            classification = 'unknown'
            dominant_color = 'none'
            dominant_percent = 0
        
        # Save visualization
        fig = visualize_wound_colors_detailed(image, mask, distributions, color_masks)
        fig.savefig(os.path.join(output_dir, f'{image_file.stem}_analysis.png'))
        plt.close(fig)
        
        # Add confidence calculation
        confidence = calculate_confidence(distributions, classification)
        
        # Store results with confidence
        results.append({
        'Image': base_name,
        'Classification': classification,
        'Confidence': round(confidence, 2),
        'Confidence_Level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
        'Dominant_Color': dominant_color,
        'Dominant_Percentage': round(dominant_percent, 2),
        'Red_Percentage': round(distributions.get('red', 0), 2),
        'Yellow_Percentage': round(distributions.get('yellow', 0), 2),
        'Pink_Percentage': round(distributions.get('pink', 0), 2),
        'Black_Percentage': round(distributions.get('black', 0), 2),
        'Unclassified_Percentage': round(100 - sum(distributions.values()), 2)
    })
        
        print(f"Processed {base_name}: {classification} (Confidence: {confidence:.2f})")
    
    # Save results to Excel
    if results:
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(output_dir, 'color_analysis.xlsx'), index=False)
        print(f"\nResults saved to {output_dir}")
    else:
        print("No images processed")
