from smart_stethoscope.ml_logic.data_loading import load_data
from smart_stethoscope.ml_logic.preprocessing import perprocess_tabular_data


def preprocessing():
    data = load_data()
    # add here a call to the preprocessor to preprocess data
    # maybe we want to save also the preprocessed data
    what_ever_comes_out = perprocess_tabular_data(data)


def train():
    pass
