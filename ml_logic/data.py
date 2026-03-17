import os
from pathlib import Path
import pandas as pd
import librosa as lb
import soundfile as sf


def preprocess_audio_data():

    def getPureSample(raw_data, start, end, sr=22050):
        """
        Takes a numpy array of audio data and spilts it using start and end arguments.

        raw_data=numpy array of audio sample
        start=time
        end=time
        sr=sampling_rate
        """
        max_ind = len(raw_data)
        start_ind = min(int(start * sr), max_ind)
        end_ind = min(int(end * sr), max_ind)
        return raw_data[start_ind:end_ind]

    def get_audio_annotation_data(raw_audio_path):
        """
        Takes the path to the raw audio annotation folder. Returns the annotations
        of all annotation files in the folder as a dataframe.

        raw_audio_path=path to the folder where all the raw audio annotation files are

        Returns:
        annotation_data = DataFrame of all audio annotations.
        """
        annotation_files = [
            file.split(".")[0]
            for file in os.listdir(raw_audio_path)
            if file.endswith(".txt")
        ]

        files_data = []
        for file in annotation_files:
            data = pd.read_csv(
                raw_audio_path + file + ".txt",
                sep="\t",
                names=["start", "end", "crackles", "weezels"],
            )
            data["filename"] = file
            files_data.append(data)
        annotation_data = pd.concat(files_data)
        annotation_data.drop(columns=["crackles", "weezels"])
        return annotation_data.reset_index()

    preproc_audio_path = "../preprocessed_data/audio_breathing_cycles/"
    raw_audio_path = "../raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"

    Path(preproc_audio_path).mkdir(parents=True, exist_ok=True)

    annotation_data = get_audio_annotation_data(raw_audio_path)

    i, c = 0, 0
    for index, row in annotation_data.iterrows():
        start = row["start"]
        end = row["end"]
        filename = row["filename"]

        audio_file_loc = raw_audio_path + filename + ".wav"

        if index > 0:
            # check if more cycles exits for same patient if so then add i to change filename
            if annotation_data.iloc[index - 1]["filename"] == filename:
                i += 1
            else:
                i = 0
        filename = filename + "_" + str(i) + ".wav"

        save_path = preproc_audio_path + filename
        c += 1

        audioArr, sampleRate = lb.load(audio_file_loc)
        pureSample = getPureSample(audioArr, start, end, sampleRate)

        sf.write(file=save_path, data=pureSample, samplerate=sampleRate)
    print("Total Files Processed: ", c)
