from smart_stethoscope.ml_logic.data_loading import load_data
from smart_stethoscope.ml_logic.preprocessing import preprocess_tabular_data


def preprocessing():
    data = load_data()
    X_train, X_test, y_train, y_test, train_pids, test_pids = preprocess_tabular_data(
        data,
        pipeline_save_path='models/post_split_pipeline.pkl'
    )


def train():
    pass

if __name__ == '__main__':
    preprocessing() #safety recommendation
