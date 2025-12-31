import os
import numpy as np

import torch
from torchvision import models
from torch import optim

from defaults import MODEL_CONFIG, TRAINING_CONFIG, GPU_DEVICE, CHECKPOINT_FILE_NAME

from image_classifier import ImageClassifier

def parse_hidden_units(value):
    """Convert comma-separated string to list of integers."""
    return [int(x.strip()) for x in value.split(',')]

def get_device(config):
    """Get the compute device based on GPU configuration."""
    use_gpu = get_config_value(config, 'gpu', False)
    return torch.device(GPU_DEVICE if use_gpu else 'cpu')

def get_config_value(config, key, default):
    """
    Get config value with proper None handling.
    
    Returns the default if the key is missing OR if the value is None.
    This prevents None values from overriding sensible defaults.
    """
    value = config.get(key)
    return default if value is None else value

def validate_checkpoint_path(checkpoint_path):
    """Validate checkpoint file path."""
    if not checkpoint_path:
        raise ValueError("Checkpoint path cannot be empty")
    if not checkpoint_path.endswith('.pth'):
        raise ValueError(f"Checkpoint must be a .pth file, got: {checkpoint_path}")
    return checkpoint_path

def validate_config(config):
    """Validate configuration values."""
    learning_rate = config.get('learning_rate')
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got: {learning_rate}")
    
    epochs = config.get('epochs')
    if epochs is not None and epochs <= 0:
        raise ValueError(f"Epochs must be positive, got: {epochs}")
    
    output_size = config.get('output_size')
    if output_size is not None and output_size <= 0:
        raise ValueError(f"Output size must be positive, got: {output_size}")
    
    return config

def load_config(args):
    """Convert command line arguments to configuration dictionary."""
    config = {
        'data_dir': getattr(args, 'data_dir', 'flower_data'),
        'save_dir': getattr(args, 'save_dir', '.'),
        'arch': getattr(args, 'arch', 'resnet50'),
        'learning_rate': getattr(args, 'learning_rate', 0.003),
        'hidden_units': getattr(args, 'hidden_units', [765]),
        'feature_size': getattr(args, 'feature_size', 2084),
        'output_size': getattr(args, 'output_size', 102),
        'epochs': getattr(args, 'epochs', 100),
        'gpu': getattr(args, 'gpu', True),
    }
    config = {k: v for k, v in config.items() if v is not None}
    return validate_config(config)

def _load_checkpoint_file(checkpoint_path, strict=False):
    """Load checkpoint from file with error handling."""
    if checkpoint_path:
        validate_checkpoint_path(checkpoint_path)
    checkpoint = {}
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if not isinstance(checkpoint, dict):
            print(f"[WARNING] :: Checkpoint is not a dictionary, initializing fresh")
            checkpoint = {}
    except FileNotFoundError:
        if not strict:
            print(
                '[INFO] :: Checkpoint file not found: '
                'initializing model with default weights'
            )
            checkpoint = {}
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
        print(f"[WARNING] :: Error loading checkpoint: {str(e)}. Initializing fresh.")
        checkpoint = {}
    return checkpoint

def _build_model_from_checkpoint(checkpoint, config):
    """Build and configure model from checkpoint."""
    base_model = get_config_value(config, 'arch', 'resnet50')
    model = models.get_model(base_model, weights='DEFAULT')
    classifier_model = get_classifier(checkpoint, config)

    freeze_model_parameters(model)
    model.fc = classifier_model

    fallback_lr = checkpoint.get(
        'learning_rate',
        TRAINING_CONFIG['learning_rate']
    )
    learning_rate = get_config_value(config, 'learning_rate', fallback_lr)

    if checkpoint and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    device = get_device(config)
    model.to(device)

    return model, optimizer

def load_checkpoint_into_model(checkpoint_path, config, strict=False):
    checkpoint = _load_checkpoint_file(checkpoint_path, strict)
    model, optimizer = _build_model_from_checkpoint(checkpoint, config)

    model.class_to_idx = checkpoint.get('class_to_idx', None)
    return model, optimizer, checkpoint

def get_classifier(checkpoint, config):
    """Create a classifier model from checkpoint or config parameters."""
    feature_size = get_config_value(
        config,
        'feature_size',
        checkpoint.get('feature_size', MODEL_CONFIG['feature_size'])
    )
    output_size = get_config_value(
        config,
        'output_size',
        checkpoint.get('output_size', MODEL_CONFIG['output_size'])
    )
    hidden_layers = get_config_value(
        config,
        'hidden_layers',
        checkpoint.get('hidden_layers', MODEL_CONFIG['hidden_layers'])
    )
    dropout = get_config_value(
        config,
        'dropout',
        checkpoint.get('dropout', TRAINING_CONFIG['dropout'])
    )
    device = get_device(config)

    classifier_model = ImageClassifier(feature_size, output_size, hidden_layers, device=device, dropout=dropout)

    return classifier_model

def freeze_model_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

def save_model_checkpoint(model, optimizer, training_data, config):
    """Save model checkpoint with training progress and hyperparameters."""
    checkpoint_data = {
        'feature_size': get_config_value(
            config,
            'feature_size',
            MODEL_CONFIG['feature_size']
        ),
        'output_size': get_config_value(
            config,
            'output_size',
            MODEL_CONFIG['output_size']
        ),
        'hidden_layers': get_config_value(
            config,
            'hidden_layers',
            MODEL_CONFIG['hidden_layers']
        ),
        'learning_rate': get_config_value(
            config,
            'learning_rate',
            TRAINING_CONFIG['learning_rate']
        ),
        'dropout': get_config_value(
            config,
            'dropout',
            TRAINING_CONFIG['dropout']
        ),
        'epochs': training_data['epochs'],
        'current_epoch': training_data['current_epoch'],
        'accuracy': training_data['accuracy'],
        'training_losses': training_data['training_losses'],
        'validation_losses': training_data['validation_losses'],
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    save_dir = get_config_value(config, 'save_dir', '.')
    save_path = os.path.join(save_dir, CHECKPOINT_FILE_NAME)
    print(f"Saving checkpoint to {save_path}")

    torch.save(checkpoint_data, save_path)

    print(f"Checkpoint saved to {save_path}")