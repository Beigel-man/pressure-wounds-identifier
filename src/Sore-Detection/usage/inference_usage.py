import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from functions.nn import get_pretrained_unet
from functions.inference import inference
from functions.post_process import post_process_mask
import torch



def main():
    # Define paths (the py folder path is relative to this file)
    py_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    unlabeled_images_dir = py_folder_path + '/data/unlabeled/images'
    output_dir = py_folder_path + '/results/predicted_masks'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = get_pretrained_unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load pretrained weights
    model.load_state_dict(torch.load(py_folder_path + '/results/trained_model.pth'))

    # Run inference (post process can be a different function)
    inference(model=model, input_dir=unlabeled_images_dir, output_dir=output_dir, post_process_mask=post_process_mask, device=device)
if __name__ == '__main__':
    main()