import argparse
from test_utils import load_config, test

def main():
    parser = argparse.ArgumentParser(description='Test a trained model on test dataset')
    parser.add_argument('data_dir', type=str, help='path to the dataset')
    parser.add_argument('checkpoint', type=str, help='path to model checkpoint file')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')

    args = parser.parse_args()
    config = load_config(args)

    test(config)

if __name__ == '__main__':
    main()
