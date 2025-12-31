import os
import torch
from torchvision import datasets, transforms
import defaults

def create_dataloader(data_dir):
    # Validate data directory structure
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    for dir_path, name in [(train_dir, 'train'), (valid_dir, 'valid'), (test_dir, 'test')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")

    batch_size = defaults.TRAINING_CONFIG['batch_size']

    training_transforms = transforms.Compose(defaults.TRAINING_TRANSFORMS)
    validation_transforms = transforms.Compose(defaults.VALIDATION_TRANSFORMS)
    testing_transforms = transforms.Compose(defaults.TESTING_TRANSFORMS)

    training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

    trainloader = torch.utils.data.DataLoader(training_datasets, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_datasets, batch_size=batch_size, shuffle=True)
    testloader =torch.utils.data.DataLoader(testing_datasets, batch_size=batch_size, shuffle=True)

    image_datasets = {
        'train': training_datasets,
        'validate': validation_datasets,
        'test': testing_datasets
    }

    image_loaders = {
        'train': trainloader,
        'validate': validloader,
        'test': testloader
    }

    return image_datasets, image_loaders

    