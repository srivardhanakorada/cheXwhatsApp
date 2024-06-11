import os
import torch
from imageio import imread
# from lung_segmentation import segmentation_func, PretrainedUNet
from cheXwhatsApp.lung_segmentation import segmentation_func, PretrainedUNet
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import sys
    
warnings.filterwarnings('ignore')
    
def load_segmentation_model(model_path, device):
    """
    Loading the lung segmentation Model
    
    inputs - 
        model_path: path to the .pt file
        device: 'cpu' or 'cuda'
        
    output -
        segmentation_model: lung segmentation model loaded on the specified device
    """
    segmentation_model = torch.load(model_path)
    segmentation_model.to(device)
    segmentation_model.eval()
    return segmentation_model


def get_dataframe(image_path, heatmap_hr_dir, heatmap_lr_dir, segmentation_model, device, image_size, thresholds):
    """
    Function to get dataframe of diceindex between hr and lr images
    
    Args:
        image_path (str): Path to the original images.
        heatmap_hr_dir (str): Directory of high-resolution heatmaps.
        heatmap_lr_dir (str): Directory of low-resolution heatmaps.
        segmentation_model: Loaded lung segmentation model.
        device: Torch device (CPU or CUDA).
        image_size (int): Size of the images.
        thresholds (list): List of thresholds.
        
    Returns:
        pd.DataFrame: DataFrame with calculated scores.
    """
    filenames = [f for f in os.listdir(image_path) if not f.startswith('.')]
    df = pd.DataFrame(columns=['Image_Name'] + [f'HR_Score_{t}' for t in thresholds] + [f'LR_Score_{t}' for t in thresholds])

    iter_filenames = tqdm(enumerate(filenames))

    for idx, fname in iter_filenames:
        row = {}
        lung_region = segmentation_func(os.path.join(image_path, fname), segmentation_model, device)
        lung_region = resize(lung_region, (image_size, image_size))
        
        heatmap_hr = imread(os.path.join(heatmap_hr_dir, fname))
        heatmap_lr = imread(os.path.join(heatmap_lr_dir, fname))
        row['Image_Name'] = fname
        
        for threshold in thresholds:
            dice_hr = (np.sum((heatmap_hr > int(threshold * 255)) * (lung_region > 0)) / 
                       (np.sum(heatmap_hr > int(threshold * 255)) + 1e-7))
            dice_lr = (np.sum((heatmap_lr > int(threshold * 255)) * (lung_region > 0)) / 
                       (np.sum(heatmap_lr > int(threshold * 255)) + 1e-7))
            row[f'HR_Score_{threshold}'] = dice_hr
            row[f'LR_Score_{threshold}'] = dice_lr

        df = df._append(row, ignore_index=True)
        
    return df


def ols_score(image_path, heatmap_hr_dir, heatmap_lr_dir, device='cpu', image_size=512, thresholds=[0.5]):
    """Function to compute OLS Score and return the results as a DataFrame
    
    Args:
        image_path (str): Path to the original images.
        output_dir (str): Directory to save the output CSV file.
        model_cam_name_hr (str): Directory for high-resolution heatmaps.
        model_cam_name_lr (str): Directory for low-resolution heatmaps.
        device (str): Device to use for computation ('cpu' or 'cuda').
        image_size (int): Size to resize the images.
        thresholds (list): List of thresholds for score calculation.
        
    Returns:
        pd.DataFrame: DataFrame containing the OLS scores.
    """
    current_file_path = __file__

    # Get absolute path (Python 3.9+ always returns absolute, earlier versions might return relative)
    absolute_path = os.path.abspath(current_file_path)

    # print(f"Current file path: {current_file_path}")
    # print(f"Absolute path: {absolute_path}")
    
    sys.path.insert(0, absolute_path.replace('ols_score.py',''))
    model_path = absolute_path.replace('ols_score.py','') + 'lung_segmentation.pt'

    print(model_path)
    segmentation_model = load_segmentation_model(model_path, device)
    
    df = get_dataframe(image_path, heatmap_hr_dir, heatmap_lr_dir, segmentation_model, device, image_size, thresholds)
    
    return df