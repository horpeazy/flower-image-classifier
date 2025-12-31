import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from dataloader import create_dataloader
from train_utils import (
    load_checkpoint_into_model,
    save_model_checkpoint,
    get_device,
    get_config_value
)

import defaults

def train(config):
    data_dir = get_config_value(config, 'data_dir', 'flower_data')
    save_dir = get_config_value(config, 'save_dir', '.')
    device = get_device(config)
    
    checkpoint_path = os.path.join(save_dir, defaults.CHECKPOINT_FILE_NAME)

    image_datasets, loaders = create_dataloader(data_dir)
    model, optimizer, checkpoint = load_checkpoint_into_model(checkpoint_path, config)
    model.class_to_idx = image_datasets['train'].class_to_idx

    criterion = nn.NLLLoss()

    training_results = train_model(model, optimizer, criterion, loaders, checkpoint, config, device=device)
    # plot_training_results(training_results)

    save_model_checkpoint(model, optimizer, training_results, config)

    # Run final validation with detailed output
    print(f"\n{'='*50}")
    print("Running final validation on validation dataset...")
    print(f"{'='*50}\n")
    validate_model(model, criterion, loaders['validate'], device, verbose=True)
    print(f"\n{'='*50}")
    print("Validation completed!")
    print(f"{'='*50}\n")

    # Run test with detailed output
    print(f"{'='*50}")
    print("Running model evaluation on test dataset...")
    print(f"{'='*50}\n")
    test_model(model, criterion, loaders, device)
    print(f"\n{'='*50}")
    print("Testing completed!")
    print(f"{'='*50}\n")

    save_model_checkpoint(model, optimizer, training_results, config)


def validate_model(model, criterion, loader, device, verbose=False):
    """
    Run validation and return average loss and accuracy.
    
    Args:
        model: The trained PyTorch model
        criterion: Loss function
        loader: Validation data loader
        device: Device to run validation on (CPU/GPU)
        verbose: If True, prints detailed progress and metrics
    
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            log_probabilities = model.forward(images)
            loss = criterion(log_probabilities, labels)
            total_loss += loss.item()
            
            probabilities = torch.exp(log_probabilities)
            _, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            total_samples += labels.size(0)
            
            # Progress indicator if verbose
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")
    
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    
    # Print detailed results if verbose
    if verbose:
        print(f"\nValidation Results:")
        print(f"  Validation Loss: {avg_loss:.4f}")
        print(f"  Validation Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"  Total samples evaluated: {total_samples}")
    
    return avg_loss, avg_accuracy

def train_model(model, optimizer, criterion, loaders, model_checkpoint=None, config=None, device=defaults.DEVICE):
    if model_checkpoint is None:
        model_checkpoint = {}
    
    epochs = model_checkpoint.get('epochs', defaults.TRAINING_CONFIG['epochs'])
    current_epoch = model_checkpoint.get('current_epoch', 0)
    best_loss = model_checkpoint.get('best_loss', float('inf'))
    validate_every = defaults.VALIDATION_FREQUENCY

    training_losses, validation_losses, accuracy = [], [], []
    stagnation_counter = 0
    patience = defaults.EARLY_STOPPING_PATIENCE
    steps = 0
    stop_training = False

    print(f"Training model on {device}")

    for epoch in range(current_epoch, epochs):
        current_epoch = epoch
        train_loss = 0

        if stop_training:
            break

        print(f"Epoch {epoch}/{epochs}... ")

        for image, label in loaders['train']:
            image, label = image.to(device), label.to(device)
            steps += 1
            log_probabilities = model.forward(image)
            loss = criterion(log_probabilities, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % validate_every == 0:
                valid_loss, validation_accuracy = validate_model(model, criterion, loaders['validate'], device)

                training_losses.append(train_loss / validate_every)
                validation_losses.append(valid_loss)
                accuracy.append(validation_accuracy)

                print(f"Training Loss: {training_losses[-1]:.3f}.. "
                      f"Validation Loss: {validation_losses[-1]:.3f}.. "
                      f"Validation Accuracy: {validation_accuracy:.3f}")

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                train_loss = 0
                model.train()

            if stagnation_counter > patience:
                print("Stopping Early")
                stop_training = True
                break

    training_data = {
        'epochs': epochs,
        'current_epoch': current_epoch,
        'accuracy': accuracy[-1],
        'training_losses': np.sum(training_losses),
        'validation_losses': np.sum(validation_losses)
    }

    print("Training completed successfully")

    return training_data

def test_model(model, criterion, loaders, device=defaults.DEVICE):
    """
    Evaluate model performance on test dataset with detailed output.
    
    Args:
        model: The trained PyTorch model
        criterion: Loss function
        loaders: Dictionary containing data loaders
        device: Device to run evaluation on (CPU/GPU)
    
    Returns:
        tuple: (average_test_loss, average_test_accuracy)
    """
    test_loss = 0
    test_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        model.eval()
        for batch_idx, (images, labels) in enumerate(loaders['test']):
            images, labels = images.to(device), labels.to(device)
            log_probabilities = model.forward(images)
            loss = criterion(log_probabilities, labels)
            test_loss += loss.item()

            probabilities = torch.exp(log_probabilities)
            _, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            total_samples += labels.size(0)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loaders['test'])} batches...")

    avg_test_loss = test_loss / len(loaders['test'])
    avg_test_accuracy = test_accuracy / len(loaders['test'])
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Accuracy: {avg_test_accuracy:.4f} ({avg_test_accuracy*100:.2f}%)")
    print(f"  Total samples evaluated: {total_samples}")
    
    return avg_test_loss, avg_test_accuracy

def plot_training_results(training_results):
    plt.plot(training_results['training_losses'], label='Training Losses')
    plt.plot(training_results['validation_losses'], label='Validation Losses')
    plt.plot(training_results['accuracy'], label='Accuracy')
    plt.legend()
    plt.show()
