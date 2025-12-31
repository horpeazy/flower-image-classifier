import torch
from torchvision import transforms

CHECKPOINT_FILE_NAME = 'checkpoint.pth'

# Image preprocessing constants
IMAGE_RESIZE_SIZE = 256
IMAGE_CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VALIDATION_FREQUENCY = 5
EARLY_STOPPING_PATIENCE = 5

SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')

DEVICE = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

GPU_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda'

MODEL_CONFIG = {
    'feature_size': 2048,
    'output_size': 102,
    'hidden_layers': [764],
}

TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.004,
    'batch_size': 64,
    'device': DEVICE,
    'dropout': 0.3
}

TRAINING_TRANSFORMS = [transforms.Resize(IMAGE_RESIZE_SIZE),
                        transforms.CenterCrop(IMAGE_CROP_SIZE),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

VALIDATION_TRANSFORMS = [transforms.Resize(IMAGE_RESIZE_SIZE),
                        transforms.CenterCrop(IMAGE_CROP_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

TESTING_TRANSFORMS = [transforms.Resize(IMAGE_RESIZE_SIZE),
                        transforms.CenterCrop(IMAGE_CROP_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
