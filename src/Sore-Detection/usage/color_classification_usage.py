import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.color_classification import process_directory

# Define paths (the py folder path is relative to this file)
py_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
input_dir = py_folder_path + '/results/predicted_masks'
output_dir = py_folder_path + '/results/color_classification'


# Run color classification

def main():
    # For process_directory function to work, first- masks should be created
    # second- the input_dir should contain both images and masks
    # third- the images should be named as <base_name>_original.jpg and masks as <base_name>_MASK.jpg (that's how they are saved after inference) 
    process_directory(input_dir=input_dir, output_dir=output_dir)

if __name__ == '__main__':
    main()