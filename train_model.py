from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.TrainingDataCreator import TrainingDataCreator

def main():

    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = TrainingDataCreator(window_length=4, prediction_horizon=3)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    labels, feature_windows, mask_windows = data_creator.create_training_data(study_df, missing_masks)

    print(labels.shape)
    print(feature_windows.shape)
    print(mask_windows.shape)

if __name__ == '__main__':
    main()
