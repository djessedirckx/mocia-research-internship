import argparse
import pathlib

from eda_preprocessing.DataPreprocessor import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset and create masks')

    # Data split
    # parser.add_argument('train_percentage', default=60, help='Percentage of data which is used for training')

    # Path variables
    parser.add_argument('--input_file', help='file containing input data')

    # Class imbalance options
    # parser.add_argument('oversample', default=False, help='Apply oversampling on training data')
    parser.add_argument('--label_forwarding', action='store_true', default=False, help='Apply label forwarding on training data')

    # Parse arguments
    args = parser.parse_args()

    data_preprocessor = DataPreprocessor(args.input_file, args.label_forwarding)
    study_df, missing_masks = data_preprocessor.preprocess_data()
    print(missing_masks.head())

if __name__ == '__main__':
    main()