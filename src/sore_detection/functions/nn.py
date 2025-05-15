import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import copy
import matplotlib.pyplot as plt
import numpy as np
from functions.post_process import post_process_mask
from scipy.ndimage import binary_erosion, distance_transform_edt


# Define the Unet model with ResNet encoder
def get_pretrained_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model

# Dice Loss # accounts for the overall shape of the GT and predicted mask
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# Combined Loss # combines BCE and Dice loss
def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target) # This loss treats each pixel independently
    dsc = dice_loss(pred, target)
    # return 0.75 * bce + 0.25 * dsc

    # Dynamic weighting based on the relative magnitudes of the losses- the bigger the loss the bigger the weight
    # This ensures that the loss functions are balanced
    # and that neither dominates the other during training.
    # This is a simple approach, and more sophisticated methods exist
    # but this should work well for this case.
    total_loss = bce + dsc
    bce_weight =  bce / total_loss
    dice_weight = dsc / total_loss

    return bce_weight * bce + dice_weight * dsc


# Training loop function

def train_model(model, train_loader, val_loader, optimizer, combined_loss, device, num_epochs=400):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)

                # Calculate validation loss
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Save the best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())


    # After training finishes


    # Load best model weights
    model.load_state_dict(best_model_wts)
    print("Training complete! Best val loss:", best_loss)
    return model, train_losses, val_losses

def visualize_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Metrics for validation
# Sensitivity, Specificity, and Mean Surface Distance (MSD)

def compute_sensitivity(pred, gt):
    pred = ~pred
    gt = ~gt
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp / (tp + fn + 1e-6)

def compute_specificity(pred, gt):
    pred = ~pred
    gt = ~gt
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return tn / (tn + fp + 1e-6)

def compute_msd(pred, gt):
    pred = ~pred
    gt = ~gt
    pred_edge = pred ^ binary_erosion(pred)
    gt_edge = gt ^ binary_erosion(gt)
    dt_gt = distance_transform_edt(~gt_edge)
    dt_pred = distance_transform_edt(~pred_edge)
    d1 = dt_gt[pred_edge].mean() if pred_edge.any() else 0
    d2 = dt_pred[gt_edge].mean() if gt_edge.any() else 0
    return (d1 + d2) / 2

# validation- find masks and plot: Raw, After Thresholding, and Post-processed

def check_validation(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # convert logits to probabilities

        # Apply dynamic thresholding and post-processing
        predicted_masks = torch.zeros_like(outputs)
        for i in range(outputs.shape[0]):
            prob_map = outputs[i][0].cpu().numpy()

            # Dynamic threshold - mean - 0.5 std
            thresh_val = prob_map.mean() - 0.5 * prob_map.std()
            binary_mask = (prob_map < thresh_val).astype(np.uint8)

            # Post-process
            post_processed = post_process_mask(binary_mask)

            # Save to tensor
            predicted_masks[i][0] = torch.from_numpy(post_processed).float()

    fig, axes = plt.subplots(len(images), 5, figsize=(20, 5*len(images)))
    metrics = []

    for i in range(len(images)):
        # Original image
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(masks[i][0].cpu(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Raw probabilities
        axes[i, 2].imshow(outputs[i][0].cpu(), cmap='gray')
        axes[i, 2].set_title('Raw Probabilities')
        axes[i, 2].axis('off')

        # Dynamic threshold mean - 0.5 std
        thresh_val = outputs[i][0].cpu().numpy().mean() - 0.5 * outputs[i][0].cpu().numpy().std()
        thresh_mask = (outputs[i][0].cpu().numpy() > thresh_val).astype(np.uint8)
        axes[i, 3].imshow(thresh_mask, cmap='gray')
        axes[i, 3].set_title('After Thresholding')
        axes[i, 3].axis('off')

        # Final post-processed

        axes[i, 4].imshow(predicted_masks[i][0].cpu(), cmap='gray')
        axes[i, 4].set_title('Post-processed')
        axes[i, 4].axis('off')


        # Calculate metrics
        gt = masks[i][0].cpu().numpy() > 0.5
        pred = predicted_masks[i][0].cpu().numpy() > 0.5
        sens = compute_sensitivity(pred, gt).item()
        spec = compute_specificity(pred, gt).item()
        msd = compute_msd(pred, gt)

        metrics.append({
            'index': i,
            'sensitivity': sens,
            'specificity': spec,
            'msd': msd
        })

        print(f"Image {i}: Sensitivity={sens:.3f}, Specificity={spec:.3f}, MSD={msd:.2f}")

    plt.tight_layout()
    plt.show()
