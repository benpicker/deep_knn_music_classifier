import librosa
import numpy as np
import os
import csv
from utils import get_project_root

# directories used to create features
root_directory = get_project_root()
original_data_directory = f"{root_directory}/data/genres/"
data_file_directory = f"{root_directory}/data/feature_data/"

def get_features(file_path):
    """
    For a given file, it computes the features from that audio wave
    """
    # return []
    y, sr = librosa.load(file_path, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    row = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        row.append(np.mean(e))
    row = np.array(row)
    return row


if __name__ == "__main__":
    # create header for features, labels, and other data
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    # open empty file and add header
    file = open(f'{data_file_directory}/data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # generate features obsevation by observation
    idx = 0
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for genre in genres:
        for filename in os.listdir(f"{original_data_directory}/{genre}"):
            file_path = f"{original_data_directory}/{genre}/{filename}"
            row = get_features(file_path)
            row = [str(num) for num in row]
            row = [filename] + row + [genre]
            file = open(f"{data_file_directory}/data.csv", 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(row)
            if (idx != 0) and (idx % 50 == 0):
                print(f"{idx} examples completed")
            idx += 1


