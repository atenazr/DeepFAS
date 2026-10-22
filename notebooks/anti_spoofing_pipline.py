# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.optim as optim
import torchinfo
from tqdm.notebook import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict
from torchvision.utils import save_image
from random import randint, randrange
import math
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Base directory --- tha last path would be test data, others train
BASE_PATH = [
'/kaggle/input/datasets/atenazare/reply-original',
            # '/kaggle/input/datasets/atenazare/msu-original'
            ] 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Computation device

WIDTH, HEIGHT = 520, 520 # Dimensions for image resizing
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # Mean and Standard Deviation for normalization
INV_MEAN, INV_STD = [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225] # Inverse Mean and Standard Deviation for denormalization

PIN_MEMORY = True if torch.cuda.is_available() else False # Pin memory for faster GPU transfer
NUM_WORKERS = os.cpu_count() # Number of workers for data loading
BATCH_SIZE=28 # Number of samples per batch

EPOCHS=20 # Number of epochs for training - deeplabv3
LEARNING_RATE=1e-4 # Learning rate for model parameter updates

TRAIN_TEST_SPLIT = 0.25 # the rate split of train and test data
MAX_NUMBER_DATA = 4200 # the whole number of data
NUM_CLASSES= 2 # number of calsses - live/spoof
CLASSIFIER_EPOCHS = 15 # number of epochs for training - classifier

OUTPUT_ROOT = "/kaggle/working/results" # in this root, generated masks, feature maps, and results will be stored

print(f'Device: {DEVICE}') # Print the chosen device

# %% [code] {"jupyter":{"outputs_hidden":false}}
class InvTransform(object):
    """
    Inverse the standardization applied to images.

    Args:
        inv_mean (list of float): Mean values used to denormalize images.
        inv_std (list of float): Standard deviation values used to denormalize images.
    """
    def __init__(self, inverse_mean, inverse_std):
        # Define the transformer
        self.inverse_transform = transforms.Normalize(
            mean=inverse_mean,
            std=inverse_std,
        )

    def __call__(self, image):
        """
        Denormalize the image to its original form.

        Args:
            image (torch.Tensor): The normalized image to be denormalized.

        Returns:
            image (torch.Tensor): The denormalized image.
        """
        image = self.inverse_transform(image)

        return image

inv_transform = InvTransform(INV_MEAN, INV_STD)

# %% [code] {"jupyter":{"outputs_hidden":false}}
class Binarize(object):
    """
    A custom transformer to binarize the masks based on a threshold.

    Args:
        threshold (float): Pixels above this number become 1, and the others become 0.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, mask):
        """
        Binarieze the mask.

        Args:
            mask (torch.Tensor or np.ndarray): The mask to be binarized.

        Returns:
            mask_bin (torch.Tensor or np.ndarray): The binarized mask.
        """
        mask_bin = (mask > self.threshold).float()
        # mask_bin = mask
        return mask_bin

binarize = Binarize(threshold=0.5)

# %% [code] {"jupyter":{"outputs_hidden":false}}
IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),    # Resize images
        transforms.ToTensor(),    # Convert nd.arrays into torch.tensors and scale the values in the range of [0, 1]
        transforms.Normalize(mean=MEAN, std=STD), # Normalize the values
    ])
    # Define a transformer for output masks
MASK_TRANSFORM = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),    # Resize masks
        transforms.ToTensor(),    # Convert nd.arrays into torch.tensors
        Binarize(threshold=0.5),    # Binarize the values
])

# %% [code] {"jupyter":{"outputs_hidden":false}}
def collect_images(root, split):
    """
        Collect images by folder.

        Args:
            root (list of strings): list of datasets paths
            split (string) : detects images belong to train set or test set
            [if we have multiple datasets, test data will be the last path and train will be others]
            
        Returns:
            groups (list of strings) : the list of folders from which images will be chosen
        """
    
    groups = defaultdict(list)

    if len(root) == 1:
        root = Path(root[0])
        for img_path in root.rglob("cropped_*.jpg"):
            folder = img_path.parent.name
            if f"_{split}_" in folder:
                groups[folder].append(img_path)
    
    else:
        if split == 'test':
            root = Path(root[-1])
            
            for img_path in root.rglob("cropped_*.jpg"):
                folder = img_path.parent.name
                groups[folder].append(img_path)
                
        else:
            i=0
            for i in range(len(root)-1):
                root = Path(root[i])
            
                for img_path in root.rglob("cropped_*.jpg"):
                    folder = img_path.parent.name
                    groups[folder].append(img_path)

    
    # Shuffle images inside each folder
    for folder in groups:
        random.shuffle(groups[folder])

    return groups

# %% [code] {"jupyter":{"outputs_hidden":false}}
def round_robin_sample(groups, target_size):
    """
    Take one image from each folder repeatedly until target_size is reached.

    Args:
        groups : the list of folders from which images will be chosen
        target_size: the number of images 
    Return:
        sampled : the list of images  paths
    """

    folders = list(groups.keys())
    sampled = []

    # Current index inside each folder
    indices = {folder: 0 for folder in folders}

    while len(sampled) < target_size:

        progress = False

        random.shuffle(folders)

        for folder in folders:

            idx = indices[folder]

            if idx < len(groups[folder]):

                sampled.append(groups[folder][idx])

                indices[folder] += 1
                progress = True

                if len(sampled) == target_size:
                    break

        # Stop if every folder is exhausted
        if not progress:
            break

    return sampled

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_groups = collect_images(BASE_PATH, "train")
test_groups = collect_images(BASE_PATH, "test")

train_files = round_robin_sample(train_groups,
                                 MAX_NUMBER_DATA*(1-TRAIN_TEST_SPLIT))

test_files = round_robin_sample(test_groups,
                                MAX_NUMBER_DATA*TRAIN_TEST_SPLIT)

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f"Train images: {len(train_files)}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f"Test images: {len(test_files)}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
class CreateDataset(Dataset):
    
    def __init__(self, files, transform=None):

        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = self.files[idx]

        image = Image.open(img_path).convert("RGB")

        folder = img_path.parent.name
        parts = folder.split("_")

        # replay_train_live_real_client001...
        label = 1 if parts[2] == "live" else 0
        label = torch.tensor(label, dtype=torch.long)

        output_root = Path(OUTPUT_ROOT)
        #  create a path based on image paths in output directory 
        parent_path=(output_root / img_path.parent.parent.parent.name/img_path.parent.parent.name/ img_path.parent.name)
        parent_path.parent.mkdir(parents=True, exist_ok=True)
            
        mask_path = parent_path / f"mask_{img_path.stem}.png"
        map_path = parent_path / f"map_{img_path.stem}.png"

        if self.transform:
            image = self.transform(image)

        return (image,label,str(img_path),str(mask_path),str(map_path))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Datasets
# ==========================================================

train_dataset = CreateDataset(train_files,IMAGE_TRANSFORM)
test_dataset = CreateDataset(test_files,IMAGE_TRANSFORM)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

len (test_loader),len(train_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
"""
------------ Phase 1 --------------
WE CREATE BINARY MASKS FOR EACH IMAGE WITH DEEPLABV3 ---- HUMAN = WHITE, BACKGROUND = BLACK 
"""

# %% [code] {"jupyter":{"outputs_hidden":false}}
model = segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to(DEVICE)
MASKS_CREATOR_MODEL = model.eval()

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_spoof_mask(path):
  spoofed_mask = Image.new(mode="1",size=(WIDTH, HEIGHT))
  os.makedirs(os.path.dirname(path), exist_ok=True)  
  spoofed_mask.save(path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def save_live_mask(output_prediction,path):
  # create a color pallette, selecting a color for each class
  palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
  colors = (colors % 255).numpy().astype("uint8")

  # plot the semantic segmentation predictions of 21 classes in each color
  r = Image.fromarray(output_prediction.byte().cpu().numpy()).resize((WIDTH,HEIGHT))
  r.putpalette(colors)

  # convert to black and white image
  threshold = 40
  image_file = r.convert('L')
  # Threshold
  image_file = image_file.point( lambda p: 255 if p > threshold else 0 )
  # To mono
  img = image_file.convert('1')
  os.makedirs(os.path.dirname(path), exist_ok=True)  
  img.save(path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_live_mask(image,mask_path):
    image = image.unsqueeze(0)
    image = image.to(DEVICE)
    with torch.no_grad():
        output = MASKS_CREATOR_MODEL(image)['out'][0]
    output_prediction = output.argmax(0)
    save_live_mask(output_prediction,mask_path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_masks(loader,spoofed_masks=False):
    for images, labels, image_paths, mask_paths, map_paths in loader:

        for image, label, mask_path in zip(images, labels, mask_paths):
            if spoofed_masks == True:
                if label.item() == 0:
                    create_live_mask(image, mask_path)
            else:
                if label.item() == 1:
                    create_live_mask(image, mask_path)
                else:
                    create_spoof_mask(mask_path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
create_masks(train_loader)

# %% [code]
create_masks(test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
"""
------------ Phase 2 --------------
WITH CREATED BINARY MASKS IN THE PREVIOUS PHASE, WE WILL FINE-TUNE DEEPLABV3 
"""

# %% [code] {"jupyter":{"outputs_hidden":false}}
def convert_to_tensor(file_path):
    image = Image.open(file_path)

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(image)
    
    return img_tensor

# %% [code] {"jupyter":{"outputs_hidden":false}}
def convert_paths_tensors(paths):
    tensors = torch.stack([
        torch.from_numpy(
            (np.array(Image.open(p).convert("L")) > 0).astype(np.int64)
        ).unsqueeze(0)   # add channel dimension
        for p in paths]).long()
    return tensors

# %% [code] {"jupyter":{"outputs_hidden":false}}
def apply_mask_overlay(image, mask, color=(255,0,0), alpha=0.65):
    """
    Overlays a binary mask on an image with a specified color and transparency.

    Args:
        image (np.ndarray): Images.
        mask (np.ndarray): Segmentation masks.
        color (tuple of int): The color for the mask overlay
        alpha (float): Transparency factor for the overlay.

    Returns:
        overlayed_image (np.ndarray): The image with the mask overlay.

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    masked = masked.filled()

    overlayed_image = cv2.addWeighted(image, 1 - alpha, masked, alpha, 0)

    return overlayed_image

# %% [code] {"jupyter":{"outputs_hidden":false}}
def plot_images_and_masks(images, masks, predicted_masks=None,mean_iou=None,masks_to_tensor=False):
    """
    Display images alongside their actual masks and, if provided, the predicted masks.

    Args:
        images (torch.Tensor): Tensor of images to be shown.
        masks (torch.Tensor): paths of masks corresponding to each image.
        predicted_masks (torch.Tensor, optional): Tensor of predicted masks corresponding to each image.
        mean_iou (float, optional): Mean Intersection over Union (IoU) score for the dataset.
    """
    num_samples = len(images) # Number of images to display (columns)
    num_rows = 3 if predicted_masks is None else 4 # Rows for images, actual masks, and (optionally) predicted masks

    # Define the figure and axes
    fig, axes = plt.subplots(num_rows, num_samples,
                             figsize=(num_samples*5, num_samples+(num_rows*2)),
                             dpi=200,
                            facecolor='#252627')

    for i in range(num_samples):
        # Transform the image for visualization
        image = inv_transform(images[i]).permute(1, 2, 0).detach().cpu().numpy()
        # Plot image
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title('Image', fontsize=20, fontweight='bold', color='white')
        axes[0, i].axis('off')

        # Transform the mask for visualization
        mask = masks[i]
        if masks_to_tensor:
            mask = convert_to_tensor(masks[i])
      
        mask = mask.permute(1, 2, 0).detach().cpu().numpy()
        # print(f"mask{i}")
        # print(np.all(mask==0))
        # Plot actual mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title('Actual Mask', fontsize=20, fontweight='bold', color='white')
        axes[1, i].axis('off')
        
        # Plot predicted mask, if available
        if predicted_masks is not None:
            # Transform the mask for visualization
            predicted_mask = predicted_masks[i].permute(1, 2, 0).detach().cpu().numpy()
            axes[2, i].imshow(predicted_mask, cmap='gray')
            axes[2, i].set_title(
                f'Predicted M - IoU: {round(float(calculate_iou(predicted_masks[i], masks[i])), 3)}',
                                 fontsize=19, fontweight='bold', color='white'
            )
            axes[2, i].axis('off')

            overlayed_image = apply_mask_overlay(image, predicted_mask)
            overlayed_image_row = 3
        else:
            overlayed_image = apply_mask_overlay(image, mask)
            overlayed_image_row = 2

        # Plot overlayed mask
        axes[overlayed_image_row, i].imshow(overlayed_image, cmap='gray')
        axes[overlayed_image_row, i].set_title('Overlayed Image', fontsize=20, fontweight='bold', color='white')
        axes[overlayed_image_row, i].axis('off')

    # Add overall title
    plt.suptitle('Images and Masks' + (f' - Mean IoU: {mean_iou}' if mean_iou != None else ''),
                 fontsize=30, fontweight='bold', color='white')
    
    version = randint(100, 999)
    plt.savefig(f"{OUTPUT_ROOT}/IMAGES-MASKS-PRETRAINED-{version}.pdf",bbox_inches="tight")
    plt.savefig(f"{OUTPUT_ROOT}/IMAGES-MASKS-PRETRAINED-{version}.png",dpi=600,bbox_inches="tight")

    plt.show() # Display the plot

# images, labels, image_paths, mask_paths, map_paths = next(iter(train_loader))

# plot_images_and_masks(images[:8],
#                       mask_paths[:8],masks_to_tensor=True)

# it does show plot --- with this error = "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.006862755..166.04646]."

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_iou(pred_mask, true_masks):
    """
    Calculate Intersection over Union (IoU) for binary masks.

    Args:
        pred_mask (torch.Tensor): Predicted binary masks.
        true_mask (torch.Tensor): Ground truth binary masks.

    Returns:
        float: IoU score.
    """

    # Compute intersection and union
    intersection = torch.sum(pred_mask * true_masks) # Element-wise multiplication
    union = torch.sum(pred_mask + true_masks) - intersection # Element-wise summation

    # Compute IoU
    iou = intersection / (union + 1e-6)  # Add a small constant to avoid division by zero

    return iou.item()

def compute_dataset_iou(model, loader, plot=False):
    """
    Compute the mean Intersection over Union (IoU) for a model over an entire dataset.
    Optionally, visualize a batch of images with their corresponding masks and predicted masks.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader for the dataset to evaluate.
        plot (bool): Whether to visualize predicted masks alongside images and true masks.

    Returns:
        if plot is false:
            float: Mean IoU score if plot is False.
    """
    model.eval() # Set the model to evaluation mode

    iou_list = []
    with torch.no_grad():
        for images, labels, image_paths, mask_paths, map_paths in loader:
            images = images.to(DEVICE) # Move images tensors to device
            
            "------open masks----------"
            masks = convert_paths_tensors(mask_paths)
            masks = masks.to(DEVICE) # Move masks tensors to device
            
            # Perform inference and binarize outputs
            outputs = model(images)
            outputs = binarize(outputs)

            # Calculate IoU for the batch and append to list
            batch_iou = calculate_iou(outputs, masks)
            iou_list.append(batch_iou)

        # Calculate mean IoU over all batches
        mean_iou = round(sum(iou_list) / len(iou_list), 4)
        if plot:
            # Get a batch of images and masks for visualization
            images, labels, image_paths, mask_paths, map_paths = next(iter(loader))
            "------open masks----------"
            masks = convert_paths_tensors(mask_paths[10:18])
            masks =  masks.to(DEVICE)
            images = images[10:18].to(DEVICE)
            
            # Perform inference and binarize outputs
            outputs = model(images)
            outputs = binarize(outputs)

            # Plot images, masks, and predictions
            plot_images_and_masks(images,
                                 masks,
                                 outputs,
                                 mean_iou,)
        else:
            return mean_iou # Return mean IoU score

# %% [code] {"jupyter":{"outputs_hidden":false}}
class PretrainedDeepLabV3(nn.Module):
    """
    Pre-trained DeepLabV3 model with a ResNet-101 backbone (Transfer Learning).
    """
    def __init__(self):
        super(PretrainedDeepLabV3, self).__init__()
        # Load the pre-trained DeepLabV3 model
        self.pretrained_model = segmentation.deeplabv3_resnet101(pretrained=True)
      
        
    def forward(self, x):
        """
        Forward pass through the model to obtain the post-processed output.
        
        Args:
            x (torch.Tensor): Input tensor of images.
            
        Returns:
            x (torch.Tensor): Binarized output where the class with index 15 is highlighted.
        """
        x = self.pretrained_model(x)['out'] # Obtain the primary output from the model
        x = torch.argmax(x, dim=1).unsqueeze(1) # Get the class with the maximum score for each pixel
        x = (x == 15).float() # Binarize the output based on class index 15
        
        return x
    
def test_deeplabv3_mdoel(loader, plot=False):
    """
    Test the pre-trained DeepLabV3 model on the given data loader.
    Optionally, plot the predictions along with images and masks.

    Args:
        loader (DataLoader): DataLoader for the dataset to be tested.
        plot (bool): Whether to visualize the predicted masks along with images and true masks.
        
    Returns:
        float: Mean IoU score if plot is False.
    """
    pretrained_model = PretrainedDeepLabV3() # Instantiate the pre-trained DeepLabV3 model
    pretrained_model.to(DEVICE) # Move the model to the specified device
    
    # Evaluate the model
    if plot:
        compute_dataset_iou(pretrained_model, loader, plot=plot)
    else:
        return compute_dataset_iou(pretrained_model, loader, plot=plot)

# print('Pre-Trained DeepLabV3 Model Test:')
# test_deeplabv3_mdoel(test_loader, plot=True)

torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false}}
class LoRAConv2d(nn.Module):
    """Pure PyTorch LoRA wrapper for Conv2d layers."""
    def __init__(self, original_conv, r=32, alpha=64):
        super(LoRAConv2d, self).__init__()
        
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if self.original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        self.r = r
        self.scaling = alpha / r
        
        self.lora_A = nn.Conv2d(
            in_channels=original_conv.in_channels, 
            out_channels=r, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=False
        )
        
        self.lora_B = nn.Conv2d(
            in_channels=r, 
            out_channels=original_conv.out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.original_conv(x) + (self.lora_B(self.lora_A(x)) * self.scaling)


class ModifiedDeepLabV3_DeepLoRA(nn.Module):
    """
    Memory-Efficient LoRA: Injects LoRA ONLY into Layer 4.
    Frees up massive amounts of VRAM by not tracking gradients in Layers 1-3.
    """
    def __init__(self, r=32, alpha=64):
        super(ModifiedDeepLabV3_DeepLoRA, self).__init__()
        
        # 1. Load Pretrained Model
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        self.model.aux_classifier = None  
        
        # 2. Freeze the ENTIRE base model first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Final classifier for Binary Segmentation (Trainable)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        
        # 4. INJECT LORA ONLY INTO LAYER 4
        # Layer 4 contains the deepest, most semantic features. 
        # By only modifying layer 4, we save Gigabytes of activation VRAM.
        self._inject_lora(self.model.backbone.layer4, r=r, alpha=alpha)
        
        self.sigmoid = nn.Sigmoid()
        
    def _inject_lora(self, module, r, alpha):
        """Recursively find and replace 'conv2' layers with LoRAConv2d."""
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and name == "conv2":
                setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha))
            else:
                self._inject_lora(child, r, alpha)

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,}")

    def forward(self, x):
        x = self.model(x)['out']
        return self.sigmoid(x)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def train_model(model, loader, loss_function, optimizer):
    """
    Train the input model for one epoch.

    Args:
        model (nn.Module): PyTorch model to be trained.
        loader (DataLoader): DataLoader that provides the training data.
        loss_function (nn.Module): Loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.

    Returns:
        running_loss (float): Total loss of all batches in the DataLoader.
    """
    model.train() # Set the model to training mode

    running_loss = 0.0 # Initialize total loss over the dataset

    train_loop = tqdm(loader, leave=True) # Initialize tqdm progress bar

    for images, labels, image_paths, mask_paths, map_paths in train_loop:
        images = images.to(DEVICE) # Move images to device
        labels = labels.long().to(DEVICE)    
        "------open masks----------"
        masks = convert_paths_tensors(mask_paths)
        
        masks = masks.float().to(DEVICE) # Move masks to device

        outputs = model(images) # Get the output
        
        #loss function
        loss = loss_function(outputs, masks)

        optimizer.zero_grad() # Zero the gradients
        loss.backward() # Backward pass to compute gradients
        optimizer.step() # Update parameters

        running_loss += loss.item() * images[0].size(0) # Accumulate loss

        train_loop.set_postfix(loss=loss.item()) # update progress bar

    return running_loss

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save the model's state to a file.

    Args:
        state (dict): Dictionary containing model state and optimizer state.
        filename (str): Name of the file to save the checkpoint.
    """
    print("===> Saving checkpoint")
    torch.save(state, f"{OUTPUT_ROOT}/{filename}")

def load_checkpoint(filename, model, optimizer):
    """
    Load the model's state from a file.

    Args:
        filename (str): Name of the file containing the checkpoint.
        model (nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer): Optimizer to load the state into.
    """
    print(f"===> Loading checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])  # Load model state
    optimizer.load_state_dict(checkpoint["optimizer"])  # Load optimizer state

# %% [code] {"jupyter":{"outputs_hidden":false}}
def fine_tune_model(model, loss_function, optimizer, train_loader, val_loader, num_epochs=EPOCHS, counter_limit=3):
    """
    Fine-tune the pre-trained DeepLabV3 model.

    Args:
        model (nn.Module): Model to be trained.
        loss_function (nn.Module): Loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
        train_loader (DataLoader): DataLoader providing training data.
        val_loader (DataLoader): DataLoader providing validation data.
        num_epochs (int): Number of epochs to train.
        counter_limit (int): Number of epochs to wait before stopping training when there is no improvement in validation IoU.

    Returns:
        history (pd.DataFrame): A DataFrame containing:
            - 'train_loss' (list of float): List of training losses per epoch.
            - 'val_iou' (list of float): List of validation IoUs per epoch.
    """
    # Lists to store training loss and validation IoU history
    train_loss_history = []
    val_iou_history = []

    # Counter for early stopping
    counter=0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs} --------------------------------------------------------------',)
        # Train the model and compute training loss
        running_loss = train_model(model, train_loader, loss_function, optimizer)
        epoch_loss = running_loss / len(train_loader.dataset) # Mean epoch loss

        # Compute validation Iou
        epoch_val_iou = compute_dataset_iou(model, val_loader)

        # Save checkpoint if there is improvement
        if epoch >= 10:
            if (all(i < epoch_val_iou for i in val_iou_history)
                or all(i > epoch_loss for i in train_loss_history)):

                # Save model and optimizer state
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }

                filename = f'Epoch-{epoch+1}.pth.tar' # Name of the checkpoint to be saved
                save_checkpoint(checkpoint, filename=filename)

            # Update counter based on validation IoU improvement
            if all(i < epoch_val_iou for i in val_iou_history):
                counter = 0
            elif counter > counter_limit:
                print(f'No improvement for {counter_limit} epochs. Stopping training.')
                break
            else:
                counter += 1

        # Store metrics
        val_iou_history.append(epoch_val_iou)
        train_loss_history.append(epoch_loss)

        print(f'Train Loss: {epoch_loss:.4f} - ',
              f'Validation IoU: {epoch_val_iou:.4f}',)

        # Check if validation IoU is between 0.95 and 0.955
        if epoch_val_iou > 0.95 and epoch_val_iou < 0.955:
            # Reduce learning rate by a factor of 0.5 for all parameter groups
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        # Check if validation IoU exceeds 0.955
        elif epoch_val_iou > 0.955:
            # Reduce learning rate by a factor of 0.1 for all parameter groups
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    # History dataframe
    history = pd.DataFrame({'train_loss': train_loss_history, 'val_iou': val_iou_history})

    # The final model is the model after the last trained epoch (no checkpoint is reloaded).

    print('Training is Done.')

    return history

deeplabv3_model = ModifiedDeepLabV3_DeepLoRA(r=32, alpha=64)
deeplabv3_model.to(DEVICE) # Move the model to device

loss_function = nn.MSELoss() # Define loss function
optimizer = optim.Adam(deeplabv3_model.parameters(), lr=LEARNING_RATE) # Define optimizer

# Get model summary
torchinfo.summary(deeplabv3_model, input_size=(1, 3, 520, 520))

# %% [code] {"jupyter":{"outputs_hidden":false}}
history = fine_tune_model(deeplabv3_model, loss_function, optimizer, train_loader, test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def plot_history():
    """
    Generates a plot comparing the training losses and validation IoU scores over epochs.

    Args:
        history (pd.DataFrame): A DataFrame containing:
            - 'train_loss' (list of float): List of training losses per epoch.
            - 'val_iou' (list of float): List of validation IoUs per epoch.

        # fig.show() # Show the plot

    history = pd.DataFrame({'train_loss': train_loss_history, 'val_iou': val_iou_history}) """
    
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_iou'], label='validation IOU', marker='o')
    history.to_csv(f"{OUTPUT_ROOT}/training.csv", index=False)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Train Loss & IoU validation over Epochs')
    plt.yscale("log")
    plt.grid(True)
    version = randint(100, 999)
    plt.savefig(f"{OUTPUT_ROOT}/loss-function-train-validation-deeplabv3.pdf",bbox_inches="tight")
    plt.savefig(f"{OUTPUT_ROOT}/loss-function-train-validation-deeplabv3.png",dpi=600,bbox_inches="tight")
    plt.show()
    
plot_history()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# load fine-tuned deeplabv3
# load_checkpoint("/kaggle/input/models/atenazare/finetuned-deeplabv3-reply-lora/pytorch/default/1/Epoch-19.pth.tar",deeplabv3_model,optimizer)

# %% [code] {"jupyter":{"outputs_hidden":false}}
fine_tuned_deeplabv3 = deeplabv3_model

# %% [code] {"jupyter":{"outputs_hidden":false}}
compute_dataset_iou(deeplabv3_model,test_loader, plot=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_auc_roc_deeplabv3(model,loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, image_paths, mask_paths, map_paths in loader:
            images = images.to(DEVICE)
            
            "------open masks----------"
            masks = convert_paths_tensors(mask_paths)
            masks = masks.to(DEVICE)
    
            outputs = model(images)  # sigmoid is already in model

            # Binarize ground truth
            masks = (masks > 0.5).float()
    
            all_preds.append(outputs.cpu().view(-1))
            all_targets.append(masks.cpu().view(-1))
    
    # Convert to NumPy arrays
    y_scores = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy().astype(int)
    
    # Check for both classes
    if len(np.unique(y_true)) < 2:
        print("Cannot compute AUC or plot ROC — only one class in y_true.")
    else:
        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
        auc_str = f"AUC: {auc:.4f}"
        with open(f"{OUTPUT_ROOT}/AUC-finetuned-deeplabv3.txt", 'w') as output:
            output.write(auc_str)
        
        print(auc_str)
    
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
        # Plot ROC
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f"{OUTPUT_ROOT}/finetuned-deeplabv3-roc.png",dpi=600,bbox_inches="tight")
        plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
calculate_auc_roc_deeplabv3(deeplabv3_model,test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
"""
------------ Phase 3 --------------
WE CREATE BINARY FEATURE MAPS FOR EACH IMAGE WITH FINE-TUNED DEEPLABV3 FROM PREVIOUS PHASE 
THEN, WE FINE-TUNED A CLASSIFIER TO PREDICT LABELS
"""

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_feature_maps(model,loader):
    
    model.eval() # Set the model to evaluation mode
    j=1
    for images, labels, image_paths, mask_paths, map_paths in loader:
        j = j+1
        num_samples = len(images) # Number of images to display (columns)
        images = images[:num_samples].to(DEVICE)

        # Perform inference and binarize outputs
        outputs = model(images)
        outputs = binarize(outputs)
        
        for i in range(num_samples): 
            
            predicted_mask = outputs[i].detach().cpu()
            # os.makedirs(os.path.dirname(map_paths[i]), exist_ok=True)  
            # predicted_mask.save(path)
            path = Path(map_paths[i])

            # Create the parent directory, not the file itself
            path.parent.mkdir(parents=True, exist_ok=True)
            
            save_image(predicted_mask, str(path))

# %% [code] {"jupyter":{"outputs_hidden":false}}
create_feature_maps(fine_tuned_deeplabv3,test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
create_feature_maps(fine_tuned_deeplabv3,train_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
class BinaryMapTransformer(nn.Module):

    def __init__(
        self,
        image_size=520,
        patch_size=20,
        in_channels=1,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_dim=256,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        assert image_size % patch_size == 0, \
            "image_size must be divisible by patch_size"

        self.num_patches_per_side = image_size // patch_size

        self.num_patches = (
            self.num_patches_per_side ** 2
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches,
                embed_dim
            ) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # MLP classification head
        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                mlp_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                mlp_dim,
                128
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        # Add positional information
        x = x + self.positional_embedding

        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)

        # Layer normalization
        x = self.norm(x)

        # Classification
        logits = self.classifier(x)


        return logits

# %% [code] {"jupyter":{"outputs_hidden":false}}
def train_classifier(model,train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=LEARNING_RATE
    )
    
    num_epochs = 15
    
    for epoch in range(num_epochs):
    
        model.train()
    
        running_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels, image_paths, mask_paths, map_paths in train_loader:
    
            maps = convert_paths_tensors(map_paths)
            maps = maps.to(DEVICE).float()
            
            labels = labels.to(DEVICE).long()
    
            optimizer.zero_grad()
    
            logits = model(maps)
    
            loss = criterion(
                logits,
                labels
            )
    
            loss.backward()
    
            optimizer.step()
    
            running_loss += loss.item()
    
            predictions = torch.argmax(logits,dim=1)
    
            correct += (
                predictions == labels
            ).sum().item()
    
            total += labels.size(0)
    
        train_accuracy = (100.0 * correct / total)
    
        average_loss = (running_loss / len(train_loader))
    
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {train_accuracy:.2f}%"
        )

# %% [code] {"jupyter":{"outputs_hidden":false}}
model = BinaryMapTransformer(
    image_size=520,
    patch_size=20,
    in_channels=1,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    mlp_dim=256,
    dropout=0.1,
    num_classes=2
).to(DEVICE)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# %% [code] {"jupyter":{"outputs_hidden":false}}
count_params(model)

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_classifier(model,train_loader, test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_metrics(model,test_loader):
    model.eval()
    
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
    
        for images, labels, image_paths, mask_paths, map_paths in test_loader:
            
            maps = convert_paths_tensors(map_paths)
            maps = maps.to(DEVICE).float()
    
            labels = labels.to(DEVICE).long()
    
            logits = model(maps)
    
            probabilities = torch.softmax(
                logits,
                dim=1
            )
    
            predictions = torch.argmax(
                logits,
                dim=1
            )
    
            correct += (
                predictions == labels
            ).sum().item()
    
            total += labels.size(0)
    
            all_labels.extend(
                labels.cpu().numpy()
            )
    
            all_predictions.extend(
                predictions.cpu().numpy()
            )
    
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )
    
    
    test_accuracy = (
        100.0 * correct / total
    )

    acc = f"Accuracy: {test_accuracy:.2f}%"
    with open(f"{OUTPUT_ROOT}/Accuracy-vit-classifier.txt", 'w') as output:
            output.write(acc)
    
    print(f"Test Accuracy:{test_accuracy:.2f}%")

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    ).ravel()
    
    
    FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
    HTER = (FAR + FRR) / 2

    output1 = "FAR : "+ str(FAR) + "| FRR :"+ str(FRR) + "| HTER : " + str(HTER)    
    with open(f"{OUTPUT_ROOT}/HTER-vit-classifier.txt", 'w') as output:
            output.write(output1)
    
    
    print(f"FAR : {FAR:.4f}")
    print(f"FRR : {FRR:.4f}")
    print(f"HTER: {HTER:.4f}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
calculate_metrics(model,test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_roc(model,test_loader): 
    model.eval()
    
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
    
        for images, labels, image_paths, mask_paths, map_paths in test_loader:
            
            maps = convert_paths_tensors(map_paths)
            maps = maps.to(DEVICE).float()
    
            labels = labels.to(DEVICE).long()
    
            logits = model(maps)
    
            probabilities = torch.softmax(
                logits,
                dim=1
            )
    
            predictions = torch.argmax(
                logits,
                dim=1
            )
    
            correct += (
                predictions == labels
            ).sum().item()
    
            total += labels.size(0)
    
            all_labels.extend(
                labels.cpu().numpy()
            )
    
            all_predictions.extend(
                predictions.cpu().numpy()
            )
    
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )
    
    # Calculate ROC curve    
    fpr, tpr, thresholds = roc_curve(
        all_labels,
        all_probabilities
    )
    
    # Calculate AUC
    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    # Plot ROC curve
    plt.figure(figsize=(7, 6))
    
    plt.plot(
        fpr,
        tpr,
        label=f"Transformer (AUC = {auc:.4f})",
        linewidth=2
    )
    
    # Random classifier
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )
    
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    
    plt.title("ROC Curve - Lightweight Transformer")
    
    plt.legend(loc="lower right")
    
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(
        "/kaggle/working/transformer_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    output1 = "AUC : "+ str(auc)    
    with open(f"{OUTPUT_ROOT}/AUC-vit-classifier.txt", 'w') as output:
            output.write(output1)
    
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
calculate_roc(model,test_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_acer(model, test_loader,threshold=0.5):
    model.eval()
    
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        
        for images, labels, image_paths, mask_paths, map_paths in test_loader:
            
            maps = convert_paths_tensors(map_paths)
            maps = maps.to(DEVICE).float()
            
            labels = labels.to(DEVICE).long()
            
            logits = model(maps)
            
            probabilities = torch.softmax(
                logits,
                dim=1
            )
            
            predictions = torch.argmax(
                logits,
                dim=1
            )
            
            correct += (
                predictions == labels
            ).sum().item()
            
            total += labels.size(0)
            
            all_labels.extend(
                labels.cpu().numpy()
            )
            
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )
    
    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Apply threshold to probability of live class
    all_predictions = (
        all_probabilities >= threshold
    ).astype(int)
    
    # --------------------------------
    # APCER
    # --------------------------------
    # Spoof/attack = 0
    # Error: spoof classified as live
    attack_samples = (
        all_labels == 0
    )
    
    apcer = np.sum(
        (all_predictions == 1) & attack_samples
    ) / np.sum(attack_samples)
    
    # --------------------------------
    # BPCER
    # --------------------------------
    # Live/bona fide = 1
    # Error: live classified as spoof
    bona_fide_samples = (
        all_labels == 1
    )
    
    bpcer = np.sum(
        (all_predictions == 0) & bona_fide_samples
    ) / np.sum(bona_fide_samples)
    
    # --------------------------------
    # ACER
    # --------------------------------
    acer = (
        apcer + bpcer
    ) / 2

    
    print(
        f"APCER : {apcer:.4f} "
        f"({apcer * 100:.2f}%)"
    )
    
    print(
        f"BPCER : {bpcer:.4f} "
        f"({bpcer * 100:.2f}%)"
    )
    
    print(
        f"ACER  : {acer:.4f} "
        f"({acer * 100:.2f}%)"
    )
    
    # Save results
    output1 = (
        f"APCER : {apcer:.6f}\n"
        f"BPCER : {bpcer:.6f}\n"
        f"ACER : {acer:.6f}\n"
    )
    
    with open(
        f"{OUTPUT_ROOT}/ACER-vit-classifier.txt",
        "w"
    ) as output:
        output.write(output1)
    
    return apcer, bpcer, acer

# %% [code] {"jupyter":{"outputs_hidden":false}}
apcer, bpcer, acer = calculate_acer(model, test_loader)
