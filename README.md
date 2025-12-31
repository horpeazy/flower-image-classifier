# Flower Image Classifier

A deep learning image classifier built with PyTorch that recognizes 102 different species of flowers. This project uses transfer learning with pre-trained convolutional neural networks and includes both a Jupyter notebook and command-line applications for training and prediction.

## Overview

This project is part of the Udacity AI Programming with Python Nanodegree program. It demonstrates the complete workflow of building an AI application:
- Loading and preprocessing image datasets
- Training an image classifier using transfer learning
- Evaluating model performance
- Using the trained model to make predictions on new images

The classifier is trained on the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the University of Oxford, containing images of flowers commonly found in the United Kingdom.

## Features

- **Transfer Learning**: Uses pre-trained ResNet50 architecture (customizable to other architectures)
- **Custom Classifier**: Flexible neural network classifier with configurable hidden layers
- **GPU Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU training
- **Checkpoint System**: Saves and resumes training progress
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Data Augmentation**: Comprehensive image transformations for better generalization
- **Command-Line Interface**: Easy-to-use scripts for training and prediction
- **Visual Predictions**: Displays predictions with probability distributions

## Requirements

### Dependencies
- Python 3.7+
- PyTorch 1.9+
- torchvision
- NumPy
- Matplotlib
- Pillow (PIL)

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow
```

## Dataset Structure

The project expects data organized in the following structure:

```
flower_data/
├── train/          # Training images
│   ├── 1/
│   ├── 2/
│   └── ...
├── valid/          # Validation images
│   ├── 1/
│   ├── 2/
│   └── ...
└── test/           # Testing images
    ├── 1/
    ├── 2/
    └── ...
```

Each numbered folder (1-102) contains images of a specific flower species. The mapping from category numbers to flower names is provided in `cat_to_name.json`.

## Project Structure

```
aipnd-project/
├── train.py                    # Command-line training script
├── predict.py                  # Command-line prediction script
├── test.py                     # Command-line testing script
├── train_utils.py              # Training utilities and checkpoint management
├── predict_utils.py            # Prediction utilities and image processing
├── test_utils.py               # Testing utilities and model evaluation
├── training.py                 # Core training and validation logic
├── dataloader.py               # Data loading and preprocessing
├── image_classifier.py         # Custom classifier architecture
├── defaults.py                 # Default configurations and constants
├── cat_to_name.json           # Flower category to name mapping
├── checkpoint.pth             # Saved model checkpoint
└── Image Classifier Project.ipynb  # Jupyter notebook for development
```

## Usage

### Training a Model

Train a new model or continue training from a checkpoint:

```bash
python train.py flower_data [OPTIONS]
```

#### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | string | required | Path to the dataset directory |
| `--save_dir` | string | `.` | Directory to save checkpoints |
| `--arch` | string | `resnet50` | Model architecture (e.g., resnet50, vgg16) |
| `--learning_rate` | float | `0.003` | Learning rate for optimizer |
| `--hidden_units` | string | `[765]` | Comma-separated list of hidden layer sizes |
| `--feature_size` | int | `2084` | Input feature size for classifier |
| `--output_size` | int | `102` | Number of output classes |
| `--epochs` | int | `100` | Number of training epochs |
| `--gpu` | flag | False | Enable GPU acceleration |

#### Training Examples

Basic training with default settings:
```bash
python train.py flower_data
```

Train with custom hyperparameters:
```bash
python train.py flower_data --learning_rate 0.001 --epochs 50 --gpu
```

Train with custom architecture:
```bash
python train.py flower_data --arch resnet50 --hidden_units 512,256,128 --gpu
```

### Making Predictions

Predict the flower species in an image:

```bash
python predict.py /path/to/image checkpoint.pth [OPTIONS]
```

#### Prediction Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `img_path` | string | required | Path to the input image |
| `checkpoint` | string | required | Path to model checkpoint file |
| `--top_k` | int | `5` | Return top K most likely classes |
| `--category_names` | string | `cat_to_name.json` | JSON file mapping categories to names |
| `--gpu` | flag | False | Use GPU for inference |

#### Prediction Examples

Basic prediction:
```bash
python predict.py flower_data/test/1/image_06743.jpg checkpoint.pth
```

Get top 3 predictions with GPU:
```bash
python predict.py flower_data/test/1/image_06743.jpg checkpoint.pth --top_k 3 --gpu
```

Use custom category names:
```bash
python predict.py image.jpg checkpoint.pth --category_names my_categories.json
```

### Testing a Model

Evaluate a trained model on the test dataset:

```bash
python test.py data_dir checkpoint.pth [OPTIONS]
```

#### Testing Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | string | required | Path to the dataset directory |
| `checkpoint` | string | required | Path to model checkpoint file |
| `--gpu` | flag | False | Use GPU for testing |

#### Testing Examples

Basic testing:
```bash
python test.py flower_data checkpoint.pth
```

Test with GPU acceleration:
```bash
python test.py flower_data checkpoint.pth --gpu
```

The test script will:
- Load the trained model from the checkpoint
- Evaluate it on the entire test dataset
- Display test loss and accuracy metrics
- Show progress during evaluation
- Provide detailed model information from the checkpoint

## Model Architecture

The classifier uses transfer learning with the following approach:

1. **Base Model**: Pre-trained CNN (default: ResNet50) trained on ImageNet
2. **Feature Extraction**: All base model parameters are frozen
3. **Custom Classifier**: Fully connected neural network replacing the final layer
   - Input: Feature vector from base model (2048 for ResNet50)
   - Hidden layers: Customizable (default: [764])
   - Dropout: 0.3 (prevents overfitting)
   - Output: 102 classes (flower species)
   - Activation: ReLU for hidden layers, Log-Softmax for output

### Training Process

- **Loss Function**: Negative Log-Likelihood Loss (NLLLoss)
- **Optimizer**: Adam with adjustable learning rate
- **Batch Size**: 64 images per batch
- **Data Augmentation**: Random rotation, horizontal/vertical flips, center cropping
- **Validation**: Performed every 5 training steps
- **Early Stopping**: Stops training if validation loss doesn't improve for 5 consecutive checks

## Image Processing

Images are preprocessed to match the requirements of ImageNet pre-trained models:

1. **Resize**: Shortest side to 256 pixels
2. **Center Crop**: 224x224 pixels
3. **Normalization**: 
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
4. **Tensor Conversion**: Convert to PyTorch tensors

## Checkpointing

The training script automatically saves checkpoints containing:
- Model state (weights and biases)
- Optimizer state
- Training progress (epoch, losses, accuracy)
- Model hyperparameters
- Class-to-index mapping

Checkpoints enable:
- Resuming interrupted training
- Using the model for predictions
- Fine-tuning with different hyperparameters

## Device Support

The project automatically detects and uses the best available compute device:

1. **Apple Silicon (MPS)**: M1/M2/M3 Macs
2. **CUDA**: NVIDIA GPUs
3. **CPU**: Fallback for systems without GPU support

Enable GPU acceleration with the `--gpu` flag in training and prediction scripts.

## Results

The trained model achieves high accuracy on the flower classification task through:
- Leveraging pre-trained ImageNet features
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Careful hyperparameter tuning

Typical performance:
- Training accuracy: ~85-95%
- Validation accuracy: ~80-90%
- Test accuracy: ~80-90%

## Jupyter Notebook

The `Image Classifier Project.ipynb` notebook provides an interactive environment for:
- Exploring the dataset
- Experimenting with model architectures
- Visualizing training progress
- Testing predictions interactively
- Prototyping new features

## Flower Categories

The model recognizes 102 flower species including:
- Pink Primrose
- Hard-leaved Pocket Orchid
- Canterbury Bells
- Sweet Pea
- English Marigold
- Tiger Lily
- Moon Orchid
- Bird of Paradise
- And 94 more species...

Full category mapping is available in `cat_to_name.json`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dataset**: [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) by Visual Geometry Group, University of Oxford
- **Course**: Udacity AI Programming with Python Nanodegree
- **Framework**: PyTorch and torchvision
- **Pre-trained Models**: PyTorch model zoo (ImageNet weights)

## Future Improvements

Potential enhancements for this project:
- Support for additional model architectures (VGG, DenseNet, EfficientNet)
- Web interface for easy image uploads and predictions
- Mobile app integration
- Multi-label classification for images with multiple flowers
- Attention mechanisms for better interpretability
- Real-time video classification
