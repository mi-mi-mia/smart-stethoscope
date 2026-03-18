from smart_stethoscope.ml_logic.data_loading import load_data
from smart_stethoscope.ml_logic.preprocessing import preprocess_tabular_data


def preprocessing():
    data = load_data()
    X_train, X_test, y_train, y_test, train_cycle_filenames, test_cycle_filenames = (
        preprocess_tabular_data(
            data, pipeline_save_path="models/post_split_pipeline.pkl"
        )
    )
    return X_train, X_test, y_train, y_test, train_cycle_filenames, test_cycle_filenames


def train():
    pass


if __name__ == "__main__":
    preprocessing()  # safety recommendation
