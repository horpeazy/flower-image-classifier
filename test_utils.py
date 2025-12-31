import torch
from torch import nn

from dataloader import create_dataloader
from train_utils import load_checkpoint_into_model, get_device, get_config_value, validate_checkpoint_path
import defaults


def load_config(args):
    """Convert command line arguments to configuration dictionary."""
    config = {
        'data_dir': getattr(args, 'data_dir', 'flower_data'),
        'checkpoint': getattr(args, 'checkpoint', None),
        'gpu': getattr(args, 'gpu', False),
    }
    config = {k: v for k, v in config.items() if v is not None}
    return validate_config(config)


def validate_config(config):
    """Validate configuration values."""
    data_dir = config.get('data_dir')
    if data_dir is None:
        raise ValueError("Data directory cannot be empty")
    
    checkpoint = config.get('checkpoint')
    if checkpoint is None:
        raise ValueError("Checkpoint file path cannot be empty")
    
    validate_checkpoint_path(checkpoint)
    
    return config


def test(config):
    """Load model and run testing on test dataset."""
    data_dir = get_config_value(config, 'data_dir', 'flower_data')
    checkpoint = config['checkpoint']
    device = get_device(config)
    
    print(f"Loading checkpoint from: {checkpoint}")
    print(f"Loading test data from: {data_dir}")
    print(f"Using device: {device}")
    
    # Load the test data
    image_datasets, loaders = create_dataloader(data_dir)
    
    # Load the model from checkpoint
    model, optimizer, checkpoint_data = load_checkpoint_into_model(checkpoint, config, strict=True)
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.to(device)
    
    # Display model information
    if checkpoint_data:
        print(f"\nModel Information:")
        print(f"  Architecture: {get_config_value(config, 'arch', 'resnet50')}")
        print(f"  Feature size: {checkpoint_data.get('feature_size', 'N/A')}")
        print(f"  Output classes: {checkpoint_data.get('output_size', 'N/A')}")
        print(f"  Hidden layers: {checkpoint_data.get('hidden_layers', 'N/A')}")
        print(f"  Training epochs completed: {checkpoint_data.get('current_epoch', 'N/A')}")
        if checkpoint_data.get('accuracy'):
            print(f"  Validation accuracy: {checkpoint_data.get('accuracy', 'N/A'):.3f}")
    
    # Create loss criterion
    criterion = nn.NLLLoss()
    
    # Run the test
    print(f"\n{'='*50}")
    print("Running model evaluation on test dataset...")
    print(f"{'='*50}\n")
    
    test_model(model, criterion, loaders, device)
    
    print(f"\n{'='*50}")
    print("Testing completed!")
    print(f"{'='*50}")


def test_model(model, criterion, loaders, device=defaults.DEVICE):
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: The trained PyTorch model
        criterion: Loss function
        loaders: Dictionary containing data loaders
        device: Device to run evaluation on (CPU/GPU)
    """
    test_loss = 0
    test_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        model.eval()
        
        for batch_idx, (images, labels) in enumerate(loaders['test']):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            log_probabilities = model.forward(images)
            loss = criterion(log_probabilities, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            probabilities = torch.exp(log_probabilities)
            _, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            total_samples += labels.size(0)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loaders['test'])} batches...")
    
    # Calculate averages
    avg_test_loss = test_loss / len(loaders['test'])
    avg_test_accuracy = test_accuracy / len(loaders['test'])
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Accuracy: {avg_test_accuracy:.4f} ({avg_test_accuracy*100:.2f}%)")
    print(f"  Total samples evaluated: {total_samples}")
    
    return avg_test_loss, avg_test_accuracy


