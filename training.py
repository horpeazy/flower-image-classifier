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

    test_model(model, criterion, loaders, device)

    save_model_checkpoint(model, optimizer, training_results, config)


def validate_model(model, criterion, loader, device):
    """Run validation and return average loss and accuracy."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            log_probabilities = model.forward(images)
            loss = criterion(log_probabilities, labels)
            total_loss += loss.item()
            
            probabilities = torch.exp(log_probabilities)
            _, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    
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
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        model.eval()
        for image, label in loaders['test']:
            image, label = image.to(device), label.to(device)
            log_probabilities = model.forward(image)
            loss = criterion(log_probabilities, label)
            test_loss += loss.item()

            probabilities = torch.exp(log_probabilities)
            _, top_class = probabilities.topk(1, dim=1)
            equals = top_class == label.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    avg_test_loss = test_loss / len(loaders['test'])
    avg_test_accuracy = test_accuracy / len(loaders['test'])
    print(f"Test Loss: {avg_test_loss:.3f}.. Test Accuracy: {avg_test_accuracy:.3f}")

def plot_training_results(training_results):
    plt.plot(training_results['training_losses'], label='Training Losses')
    plt.plot(training_results['validation_losses'], label='Validation Losses')
    plt.plot(training_results['accuracy'], label='Accuracy')
    plt.legend()
    plt.show()
