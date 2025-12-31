import argparse
from predict_utils import load_config, predict

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('img_path', type=str, help='path to the dataset')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint file')
    parser.add_argument('--top_k', type=str, help='Number of top predictions to return')
    parser.add_argument('--category_names',type=str, help='JSON file containing the class to name mapping')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')

    args = parser.parse_args()
    config = load_config(args)

    predict(config)

if __name__ == '__main__':
    main()
