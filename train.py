import argparse
from train_utils import load_config, parse_hidden_units
from training import train

def main():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('data_dir', type=str, help='path to the dataset')
    parser.add_argument('--save_dir', type=str, help='path to save the checkpoint')
    parser.add_argument('--arch', type=str, help='architecture of the model')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument(
        '--hidden_units',
        type=parse_hidden_units,
        help='comma separated list of hidden units (e.g., "512,256,128")'
    )
    parser.add_argument('--feature_size', type=int, help='feature size of the classifier model')
    parser.add_argument('--output_size', type=int, help='output size of the model')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')

    args = parser.parse_args()
    config = load_config(args)

    train(config)

if __name__ == '__main__':
    main()
