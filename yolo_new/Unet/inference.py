import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from carvana_dataset import CarvanaDataset
from unet import UNet

def pred_show_image_grid(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
       fig.add_subplot(3, len(image_dataset), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.show()


def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    # SINGLE_IMG_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/data/train/image9_aug_1.jpg"
    SINGLE_IMG_PATH = r'C:/Users/bozai/Downloads/Segmentation_from_scratch/edited/frame_20250805_212342_crop_1_DUT_96.jpg'
    DATA_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/data"
    MODEL_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/models/unet_final_3.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)

# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
# import random
# from torch.utils.data import Subset
# import numpy as np
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# from carvana_dataset import CarvanaDataset
# from unet import UNet

# def calculate_iou(pred_mask, true_mask, threshold=0.5):
#     """Calculate Intersection over Union (IoU) for binary masks"""
#     # Convert to binary masks
#     pred_binary = (pred_mask > threshold).float()
#     true_binary = (true_mask > threshold).float()
    
#     # Calculate intersection and union
#     intersection = torch.sum(pred_binary * true_binary)
#     union = torch.sum(pred_binary) + torch.sum(true_binary) - intersection
    
#     # Avoid division by zero
#     if union == 0:
#         return 1.0 if intersection == 0 else 0.0
    
#     iou = intersection / union
#     return iou.item()

# def calculate_dice_coefficient(pred_mask, true_mask, threshold=0.5):
#     """Calculate Dice coefficient for binary masks"""
#     pred_binary = (pred_mask > threshold).float()
#     true_binary = (true_mask > threshold).float()
    
#     intersection = torch.sum(pred_binary * true_binary)
#     dice = (2.0 * intersection) / (torch.sum(pred_binary) + torch.sum(true_binary))
    
#     return dice.item()

# def calculate_pixel_accuracy(pred_mask, true_mask, threshold=0.5):
#     """Calculate pixel accuracy"""
#     pred_binary = (pred_mask > threshold).float()
#     true_binary = (true_mask > threshold).float()
    
#     correct_pixels = torch.sum(pred_binary == true_binary)
#     total_pixels = torch.numel(pred_binary)
    
#     accuracy = correct_pixels / total_pixels
#     return accuracy.item()

# def evaluate_model_comprehensive(data_path, model_pth, device, num_samples=50):
#     """Comprehensive model evaluation with multiple metrics"""
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
#     model.eval()
    
#     # Load the full dataset
#     full_dataset = CarvanaDataset(data_path, test=True)
    
#     # Get random indices
#     total_images = len(full_dataset)
#     if total_images < num_samples:
#         print(f"Warning: Only {total_images} images available, using all of them.")
#         num_samples = total_images
    
#     random_indices = random.sample(range(total_images), num_samples)
#     image_dataset = Subset(full_dataset, random_indices)
    
#     print(f"Evaluating {num_samples} random images out of {total_images} total images")
    
#     # Metrics storage
#     iou_scores = []
#     dice_scores = []
#     pixel_accuracies = []
    
#     with torch.no_grad():
#         for i, (img, orig_mask) in enumerate(image_dataset):
#             img = img.float().to(device)
#             img = img.unsqueeze(0)
            
#             pred_mask = model(img)
#             pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid for binary classification
            
#             # Convert to same format for metric calculation
#             pred_mask_cpu = pred_mask.squeeze(0).cpu()
#             orig_mask_cpu = orig_mask.cpu()
            
#             # Calculate metrics
#             iou = calculate_iou(pred_mask_cpu, orig_mask_cpu)
#             dice = calculate_dice_coefficient(pred_mask_cpu, orig_mask_cpu)
#             pixel_acc = calculate_pixel_accuracy(pred_mask_cpu, orig_mask_cpu)
            
#             iou_scores.append(iou)
#             dice_scores.append(dice)
#             pixel_accuracies.append(pixel_acc)
            
#             if (i + 1) % 10 == 0:
#                 print(f"Processed {i + 1}/{num_samples} images")
    
#     # Calculate statistics
#     results = {
#         'iou': {
#             'scores': iou_scores,
#             'mean': np.mean(iou_scores),
#             'std': np.std(iou_scores),
#             'median': np.median(iou_scores),
#             'min': np.min(iou_scores),
#             'max': np.max(iou_scores)
#         },
#         'dice': {
#             'scores': dice_scores,
#             'mean': np.mean(dice_scores),
#             'std': np.std(dice_scores),
#             'median': np.median(dice_scores),
#             'min': np.min(dice_scores),
#             'max': np.max(dice_scores)
#         },
#         'pixel_accuracy': {
#             'scores': pixel_accuracies,
#             'mean': np.mean(pixel_accuracies),
#             'std': np.std(pixel_accuracies),
#             'median': np.median(pixel_accuracies),
#             'min': np.min(pixel_accuracies),
#             'max': np.max(pixel_accuracies)
#         }
#     }
    
#     return results

# def plot_evaluation_results(results):
#     """Create comprehensive visualization of evaluation results"""
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # Color palette
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
#     # Plot 1: Box plots for all metrics
#     metrics_data = [results['iou']['scores'], results['dice']['scores'], results['pixel_accuracy']['scores']]
#     metric_names = ['IoU', 'Dice Coefficient', 'Pixel Accuracy']
    
#     bp = axes[0, 0].boxplot(metrics_data, labels=metric_names, patch_artist=True)
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
    
#     axes[0, 0].set_title('Distribution of Evaluation Metrics', fontsize=14, fontweight='bold')
#     axes[0, 0].set_ylabel('Score')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Plot 2: Histogram of IoU scores
#     axes[0, 1].hist(results['iou']['scores'], bins=20, color=colors[0], alpha=0.7, edgecolor='black')
#     axes[0, 1].axvline(results['iou']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['iou']['mean']:.3f}")
#     axes[0, 1].axvline(results['iou']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['iou']['median']:.3f}")
#     axes[0, 1].set_title('IoU Score Distribution', fontsize=14, fontweight='bold')
#     axes[0, 1].set_xlabel('IoU Score')
#     axes[0, 1].set_ylabel('Frequency')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # Plot 3: Histogram of Dice scores
#     axes[0, 2].hist(results['dice']['scores'], bins=20, color=colors[1], alpha=0.7, edgecolor='black')
#     axes[0, 2].axvline(results['dice']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['dice']['mean']:.3f}")
#     axes[0, 2].axvline(results['dice']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['dice']['median']:.3f}")
#     axes[0, 2].set_title('Dice Coefficient Distribution', fontsize=14, fontweight='bold')
#     axes[0, 2].set_xlabel('Dice Score')
#     axes[0, 2].set_ylabel('Frequency')
#     axes[0, 2].legend()
#     axes[0, 2].grid(True, alpha=0.3)
    
#     # Plot 4: Scatter plot IoU vs Dice
#     axes[1, 0].scatter(results['iou']['scores'], results['dice']['scores'], alpha=0.6, color=colors[2])
#     axes[1, 0].set_xlabel('IoU Score')
#     axes[1, 0].set_ylabel('Dice Score')
#     axes[1, 0].set_title('IoU vs Dice Coefficient', fontsize=14, fontweight='bold')
    
#     # Add correlation coefficient
#     correlation = np.corrcoef(results['iou']['scores'], results['dice']['scores'])[0, 1]
#     axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[1, 0].transAxes, 
#                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # Plot 5: Performance summary bar chart
#     metrics = ['IoU', 'Dice', 'Pixel Accuracy']
#     means = [results['iou']['mean'], results['dice']['mean'], results['pixel_accuracy']['mean']]
#     stds = [results['iou']['std'], results['dice']['std'], results['pixel_accuracy']['std']]
    
#     bars = axes[1, 1].bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
#     axes[1, 1].set_title('Mean Performance with Standard Deviation', fontsize=14, fontweight='bold')
#     axes[1, 1].set_ylabel('Score')
#     axes[1, 1].set_ylim(0, 1)
    
#     # Add value labels on bars
#     for bar, mean, std in zip(bars, means, stds):
#         height = bar.get_height()
#         axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
#                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
#     axes[1, 1].grid(True, alpha=0.3)
    
#     # Plot 6: Performance over samples (trend analysis)
#     sample_indices = list(range(1, len(results['iou']['scores']) + 1))
    
#     # Calculate rolling averages
#     window_size = min(10, len(results['iou']['scores']) // 3)
#     if window_size > 1:
#         rolling_iou = np.convolve(results['iou']['scores'], np.ones(window_size)/window_size, mode='valid')
#         rolling_indices = sample_indices[window_size-1:]
#         axes[1, 2].plot(rolling_indices, rolling_iou, color=colors[0], linewidth=2, label=f'Rolling IoU (window={window_size})')
    
#     axes[1, 2].scatter(sample_indices, results['iou']['scores'], alpha=0.3, color=colors[0], s=20)
#     axes[1, 2].axhline(results['iou']['mean'], color='red', linestyle='--', alpha=0.7, label=f"Mean IoU: {results['iou']['mean']:.3f}")
#     axes[1, 2].set_xlabel('Sample Index')
#     axes[1, 2].set_ylabel('IoU Score')
#     axes[1, 2].set_title('IoU Performance Trend', fontsize=14, fontweight='bold')
#     axes[1, 2].legend()
#     axes[1, 2].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# def print_evaluation_summary(results):
#     """Print detailed evaluation summary"""
#     print("\n" + "="*60)
#     print("SEGMENTATION MODEL EVALUATION SUMMARY")
#     print("="*60)
    
#     for metric_name, metric_data in results.items():
#         print(f"\n{metric_name.upper().replace('_', ' ')}:")
#         print(f"  Mean:     {metric_data['mean']:.4f}")
#         print(f"  Median:   {metric_data['median']:.4f}")
#         print(f"  Std Dev:  {metric_data['std']:.4f}")
#         print(f"  Min:      {metric_data['min']:.4f}")
#         print(f"  Max:      {metric_data['max']:.4f}")
    
#     print("\n" + "="*60)
#     print("PERFORMANCE INTERPRETATION:")
#     print("="*60)
    
#     iou_mean = results['iou']['mean']
#     if iou_mean >= 0.9:
#         print("🟢 EXCELLENT: IoU > 0.9 - Outstanding segmentation performance")
#     elif iou_mean >= 0.7:
#         print("🟡 GOOD: IoU 0.7-0.9 - Good segmentation performance")
#     elif iou_mean >= 0.5:
#         print("🟠 FAIR: IoU 0.5-0.7 - Moderate segmentation performance")
#     else:
#         print("🔴 POOR: IoU < 0.5 - Poor segmentation performance")
    
#     print(f"\nNumber of samples evaluated: {len(results['iou']['scores'])}")
#     print("="*60)

# def pred_show_image_grid_with_iou(data_path, model_pth, device, num_samples=5):
#     """Enhanced visualization function with IoU scores"""
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
#     model.eval()
    
#     # Load the full dataset
#     full_dataset = CarvanaDataset(data_path, test=True)
    
#     # Get random indices
#     total_images = len(full_dataset)
#     if total_images < num_samples:
#         print(f"Warning: Only {total_images} images available, using all of them.")
#         num_samples = total_images
    
#     random_indices = random.sample(range(total_images), num_samples)
#     image_dataset = Subset(full_dataset, random_indices)
    
#     print(f"Processing {num_samples} random images out of {total_images} total images")
    
#     images = []
#     orig_masks = []
#     pred_masks = []
#     iou_scores = []
    
#     with torch.no_grad():
#         for img, orig_mask in image_dataset:
#             img = img.float().to(device)
#             img = img.unsqueeze(0)
            
#             pred_mask = model(img)
#             pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid
            
#             # Calculate IoU
#             iou = calculate_iou(pred_mask.squeeze(0).cpu(), orig_mask.cpu())
#             iou_scores.append(iou)
            
#             # Prepare for visualization
#             img = img.squeeze(0).cpu().detach()
#             img = img.permute(1, 2, 0)
            
#             pred_mask = pred_mask.squeeze(0).cpu().detach()
#             pred_mask = pred_mask.permute(1, 2, 0)
#             pred_mask[pred_mask < 0.5] = 0
#             pred_mask[pred_mask >= 0.5] = 1
            
#             orig_mask = orig_mask.cpu().detach()
#             orig_mask = orig_mask.permute(1, 2, 0)
            
#             images.append(img)
#             orig_masks.append(orig_mask)
#             pred_masks.append(pred_mask)
    
#     # Create the plot
#     fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 4, 12))
    
#     # Handle case when num_samples = 1
#     if num_samples == 1:
#         axes = axes.reshape(-1, 1)
    
#     for i in range(num_samples):
#         # Original images
#         axes[0, i].imshow(images[i])
#         axes[0, i].set_title(f'Original Image {i+1}')
#         axes[0, i].axis('off')
        
#         # Original masks
#         axes[1, i].imshow(orig_masks[i], cmap='gray')
#         axes[1, i].set_title(f'Ground Truth Mask {i+1}')
#         axes[1, i].axis('off')
        
#         # Predicted masks with IoU scores
#         axes[2, i].imshow(pred_masks[i], cmap='gray')
#         axes[2, i].set_title(f'Predicted Mask {i+1}\nIoU: {iou_scores[i]:.3f}')
#         axes[2, i].axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print IoU summary
#     print(f"\nIoU Scores Summary:")
#     print(f"Mean IoU: {np.mean(iou_scores):.4f}")
#     print(f"Std IoU:  {np.std(iou_scores):.4f}")
#     print(f"Min IoU:  {np.min(iou_scores):.4f}")
#     print(f"Max IoU:  {np.max(iou_scores):.4f}")

# if __name__ == "__main__":
#     DATA_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/data"
#     MODEL_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/models/unet_final.pth"
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Comprehensive evaluation
#     print("Starting comprehensive model evaluation...")
#     results = evaluate_model_comprehensive(DATA_PATH, MODEL_PATH, device, num_samples=100)
    
#     # Print summary
#     print_evaluation_summary(results)
    
#     # Plot results
#     plot_evaluation_results(results)
    
#     # Show sample images with IoU scores
#     print("\nShowing sample predictions with IoU scores...")
#     pred_show_image_grid_with_iou(DATA_PATH, MODEL_PATH, device, num_samples=10)

# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
# import random
# import numpy as np
# import os
# import glob

# from unet import UNet

# def load_images_from_folder(folder_path):
#     """Load all images from the specified folder"""
#     # Get all image files (jpg, png, etc.)
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
#     image_files = []
    
#     for extension in image_extensions:
#         image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
#     return sorted(image_files)

# def preprocess_image(image_path, input_size=(128, 128)):
#     """Preprocess image for model input"""
#     image = Image.open(image_path).convert('RGB')
    
#     transform = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.ToTensor(),
#     ])
    
#     return transform(image)

# def calculate_dice_coefficient(pred_mask, threshold=0.5):
#     """Calculate Dice coefficient for predicted mask (without ground truth)"""
#     pred_binary = (pred_mask > threshold).float()
    
#     # For visualization purposes, we'll just return the proportion of positive predictions
#     positive_ratio = torch.sum(pred_binary) / pred_binary.numel()
#     return positive_ratio.item()

# def show_predictions_only(image_folder, model_path, device, num_samples=12, input_size=(128, 128)):
#     """Show original images and their predicted masks only"""
#     # Load model
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
#     model.eval()
    
#     # Load all images from folder
#     image_files = load_images_from_folder(image_folder)
    
#     if not image_files:
#         print(f"No images found in {image_folder}")
#         return
    
#     total_images = len(image_files)
#     print(f"Found {total_images} images in the folder")
    
#     # Select random images or use all if fewer than requested
#     if total_images < num_samples:
#         print(f"Only {total_images} images available, using all of them.")
#         selected_files = image_files
#         num_samples = total_images
#     else:
#         selected_files = random.sample(image_files, num_samples)
    
#     print(f"Processing {num_samples} images...")
    
#     # Process images
#     original_images = []
#     predicted_masks = []
#     image_names = []
#     prediction_scores = []
    
#     with torch.no_grad():
#         for i, image_file in enumerate(selected_files):
#             # Load and preprocess image
#             img_tensor = preprocess_image(image_file, input_size)
#             img_tensor = img_tensor.unsqueeze(0).to(device)
            
#             # Get prediction
#             pred_mask = model(img_tensor)
#             pred_mask = torch.sigmoid(pred_mask)
            
#             # Calculate prediction score (proportion of positive pixels)
#             score = calculate_dice_coefficient(pred_mask.squeeze(0).cpu())
#             prediction_scores.append(score)
            
#             # Prepare for visualization
#             # Original image (resize for consistent display)
#             original_img = Image.open(image_file).convert('RGB')
#             original_img = original_img.resize((256, 256))  # Resize for consistent display
#             original_images.append(np.array(original_img))
            
#             # Predicted mask
#             pred_mask_np = pred_mask.squeeze(0).cpu().numpy().squeeze()
#             predicted_masks.append(pred_mask_np)
            
#             # Get image name
#             image_names.append(os.path.basename(image_file))
            
#             print(f"Processed {i+1}/{num_samples}: {os.path.basename(image_file)}")
    
#     # Create visualization
#     rows = 2  # Original image and predicted mask
#     cols = min(num_samples, 6)  # Maximum 6 columns
#     num_plots = (num_samples + cols - 1) // cols  # Number of plot grids needed
    
#     for plot_idx in range(num_plots):
#         start_idx = plot_idx * cols
#         end_idx = min(start_idx + cols, num_samples)
#         current_samples = end_idx - start_idx
        
#         fig, axes = plt.subplots(rows, current_samples, figsize=(current_samples * 3, 6))
        
#         # Handle case when current_samples = 1
#         if current_samples == 1:
#             axes = axes.reshape(-1, 1)
        
#         for i in range(current_samples):
#             idx = start_idx + i
            
#             # Original image
#             axes[0, i].imshow(original_images[idx])
#             axes[0, i].set_title(f'Original Image\n{image_names[idx]}', fontsize=10, fontweight='bold')
#             axes[0, i].axis('off')
            
#             # Predicted mask
#             axes[1, i].imshow(predicted_masks[idx], cmap='gray', vmin=0, vmax=1)
            
#             # Color code based on prediction confidence
#             score = prediction_scores[idx]
#             if score >= 0.3:
#                 color = 'green'
#                 confidence = 'High'
#             elif score >= 0.1:
#                 color = 'orange'
#                 confidence = 'Medium'
#             else:
#                 color = 'red'
#                 confidence = 'Low'
            
#             axes[1, i].set_title(f'Predicted Mask\nConfidence: {confidence} ({score:.3f})', 
#                                 fontsize=10, color=color, fontweight='bold')
#             axes[1, i].axis('off')
        
#         plt.suptitle(f'Model Predictions - Batch {plot_idx + 1}', fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.show()
    
#     # Print summary
#     print(f"\n{'='*60}")
#     print("PREDICTION SUMMARY")
#     print(f"{'='*60}")
#     print(f"Total images processed: {num_samples}")
#     print(f"Average prediction score: {np.mean(prediction_scores):.4f}")
#     print(f"Std deviation: {np.std(prediction_scores):.4f}")
#     print(f"Min score: {np.min(prediction_scores):.4f}")
#     print(f"Max score: {np.max(prediction_scores):.4f}")
    
#     # Individual scores
#     print(f"\nIndividual Prediction Scores:")
#     print("-" * 50)
#     for i, (name, score) in enumerate(zip(image_names, prediction_scores)):
#         confidence = "High" if score >= 0.3 else "Medium" if score >= 0.1 else "Low"
#         print(f"{i+1:2d}. {name:<25} Score: {score:.4f} ({confidence})")
#     print("-" * 50)

# def show_single_prediction(image_path, model_path, device, input_size=(128, 128)):
#     """Show prediction for a single image"""
#     # Load model
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
#     model.eval()
    
#     print(f"Processing single image: {os.path.basename(image_path)}")
    
#     with torch.no_grad():
#         # Load and preprocess image
#         img_tensor = preprocess_image(image_path, input_size)
#         img_tensor = img_tensor.unsqueeze(0).to(device)
        
#         # Get prediction
#         pred_mask = model(img_tensor)
#         pred_mask = torch.sigmoid(pred_mask)
        
#         # Calculate prediction score
#         score = calculate_dice_coefficient(pred_mask.squeeze(0).cpu())
        
#         # Prepare for visualization
#         original_img = Image.open(image_path).convert('RGB')
#         original_img = original_img.resize((512, 512))
        
#         pred_mask_np = pred_mask.squeeze(0).cpu().numpy().squeeze()
        
#         # Create visualization
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
#         # Original image
#         axes[0].imshow(original_img)
#         axes[0].set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=12, fontweight='bold')
#         axes[0].axis('off')
        
#         # Predicted mask
#         axes[1].imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
        
#         confidence = "High" if score >= 0.3 else "Medium" if score >= 0.1 else "Low"
#         color = 'green' if score >= 0.3 else 'orange' if score >= 0.1 else 'red'
        
#         axes[1].set_title(f'Predicted Mask\nConfidence: {confidence} (Score: {score:.4f})', 
#                          fontsize=12, color=color, fontweight='bold')
#         axes[1].axis('off')
        
#         plt.tight_layout()
#         plt.show()
        
#         print(f"Prediction score: {score:.4f} ({confidence} confidence)")

# if __name__ == "__main__":
#     IMAGE_FOLDER = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/saved_images"
#     MODEL_PATH = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/src/models/unet_final.pth"
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Show predictions for multiple images
#     print("Showing predictions for images from saved_images folder...")
#     show_predictions_only(IMAGE_FOLDER, MODEL_PATH, device, num_samples=34)
    
#     # Uncomment below to show prediction for a single specific image
#     # single_image_path = r"C:/Users/bozai/Downloads/Segmentation_from_scratch/saved_images/frame_20241225_143022.jpg"
#     # show_single_prediction(single_image_path, MODEL_PATH, device)