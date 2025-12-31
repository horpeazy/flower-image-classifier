import os
import json

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from train_utils import get_device, load_checkpoint_into_model, get_config_value
from defaults import IMAGENET_MEAN, IMAGENET_STD, SUPPORTED_IMAGE_FORMATS


def load_config(args):
    """Convert command line arguments to configuration dictionary."""
    config = {
        'img_path': getattr(args, 'img_path', None),
        'checkpoint': getattr(args, 'checkpoint', None),
        'top_k': getattr(args, 'top_k', 5),
        'category_names': getattr(args, 'category_names', 'cat_to_name.json'),
        'gpu': getattr(args, 'gpu', False),
    }
    config = {k: v for k, v in config.items() if v is not None}
    return validate_config(config)

def validate_config(config):
    """Validate configuration values."""
    img_path = config.get('img_path')
    if img_path is None:
        raise ValueError(f"Image path cannot be empty")
    
    checkpoint = config.get('checkpoint')
    if checkpoint is None:
        raise ValueError(f"Checkpoint file not found: {checkpoint}")
    
    return config

def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not image_path.lower().endswith(SUPPORTED_IMAGE_FORMATS):
        raise ValueError(f"Invalid image format. Supported: {SUPPORTED_IMAGE_FORMATS}")
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {str(e)}")

    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((height * 256) / width)
    else:
        new_height = 256
        new_width = int((width * 256) / height)
    
    image = image.resize((new_width, new_height))
    
    width, height = image.size  # Get new dimensions after resize
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image = np_image / 255.0
    np_image = np_image.transpose(2, 0, 1)
    
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    np_image = (np_image - mean) / std

    tensor_image = torch.tensor(np_image, dtype=torch.float32)
    
    return tensor_image

def show_image_from_path(image_path, ax=None, title=None):
    """Load, process, and display an image from file path."""
    if ax is None:
        _, ax = plt.subplots()
    
    processed_image = process_image(image_path)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = processed_image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def get_model_prediction(config, model, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        model.eval()

        image_path = config['img_path']
        image_tensor = process_image(image_path)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)
        logps = model.forward(image_tensor)

        ps = torch.exp(logps)
        topk = int(config.get('top_k', 5))
        top_ps, top_class = ps.topk(topk, dim=1)

    return top_ps[0], top_class[0]

def predict(config):
    checkpoint = config['checkpoint']
    image = config['img_path']
    category_names = config.get('category_names', 'cat_to_name.json')

    model, optimizer, _ = load_checkpoint_into_model(checkpoint, config, strict=True)
    device = get_device(config)
    model.to(device)

    top_ps, top_class = get_model_prediction(config, model, device)
    display_result(image, model, top_ps, top_class, category_names)


def get_class_name(idx, class_to_idx, category_names=None):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_idx = idx_to_class[idx.item()]

    if category_names:
        cat_to_name = load_category_names(category_names)
        return cat_to_name[class_idx]

    return class_idx
    
def display_result(image, model, top_ps, top_class, category_names=None):
    class_names = [get_class_name(class_idx, model.class_to_idx, category_names) for class_idx in top_class]
    _, (ax1, ax2) = plt.subplots(figsize=(10,4), ncols=2)

    show_image_from_path(image, ax2)
    ax2.axis('off')

    top_ps = top_ps.cpu().numpy()
    top_class = top_class.cpu().numpy()

    y_pos = range(len(class_names))
    ax1.barh(y_pos, top_ps)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(class_names, fontsize=10)
    
    ax1.set_xlabel('Probability')
    ax1.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()
