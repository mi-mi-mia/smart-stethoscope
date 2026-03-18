# main.py
#imports
from data import DataLoader, stratified_group_split, extract_breathing_cycles
from preprocessing import FeatureEncoder, FeatureConstructor, Imputer, Scaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# paths
audio_path = '/Users/lilyeastwood/code/mi-mi-mia/smart-stethoscope/raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
demographic_path = '/Users/lilyeastwood/code/mi-mi-mia/smart-stethoscope/raw_data/demographic_info.txt'
diagnosis_path = '/Users/lilyeastwood/code/mi-mi-mia/smart-stethoscope/raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
preproc_audio_path = '/Users/lilyeastwood/code/mi-mi-mia/smart-stethoscope/raw_data/processed_audio_files/'

# step 1 - run once to produce processed audio files
if not any(Path(preproc_audio_path).glob("*.wav")):
    extract_breathing_cycles(audio_path, preproc_audio_path)
else:
    print("✅ Processed audio files already exist, skipping extraction")

# step 2 - pre split pipeline
pre_split_pipeline = Pipeline([
    ('load', DataLoader(audio_path, demographic_path, diagnosis_path)),
    ('encode', FeatureEncoder()),
    ('construct', FeatureConstructor()),
])

data = pre_split_pipeline.fit_transform(None)

# step 3 - stratified group split
train_data, test_data = stratified_group_split(data)

# step 4 - separate X and y
le = LabelEncoder()

X_train = train_data.drop(columns=['disease'])
y_train = le.fit_transform(train_data['disease'])

X_test = test_data.drop(columns=['disease'])
y_test = le.transform(test_data['disease'])  # from test_data, not X_test

# step 5 - post split pipeline - fit on train only
post_split_pipeline = Pipeline([
    ('impute', Imputer()),
    ('scale', Scaler()),
])

X_train = post_split_pipeline.fit_transform(X_train)
X_test = post_split_pipeline.transform(X_test)

print("✅ Pipeline complete")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
