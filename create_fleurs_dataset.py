"""
many speech to text model like seamless4t require finetune on a dataset like google fleurs
from a csv file have 2 columns audio and sentence, where audio is the path to wav file (16khz, mono channel) and sentence is the transcription
create a dataset similar to google fleurs and upload it to huggingface
"""

"""
make sure load wav file into item["audio"]["array"] as numpy array like google fleurs, you can use librosa.load and np.float64 , and get same result
make sure item["audio"]["sampling_rate"] = 16000 already
"""

import pandas as pd
import uuid 
import os
import numpy as np
import librosa
def wav_to_array(wav_path):
    audio_array, sr = librosa.load(wav_path, sr=None, dtype=np.float64)

    return audio_array, sr

def fix_audio_path(audio_path):
    # file_name = os.path.basename(audio_path)
    # parent_dir = os.path.dirname(audio_path)
    return audio_path


ds_train = []
ds_val = []
train_csv = "train_linux.csv"
val_csv = "val_linux.csv"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

from tqdm import tqdm
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    id = str(uuid.uuid4())
    audio_path = row["audio"]
    transcription = row["sentence"]   
    audio_array, sampling_rate = wav_to_array(audio_path)
    ds_item = {
        "path": fix_audio_path(audio_path),
        "id": id,
        "transcription": transcription,
        "audio": {
            "path": fix_audio_path(audio_path),
            "array": audio_array,
            "sampling_rate": sampling_rate
        }
    } 
    ds_train.append(ds_item)

for index, row in val_df.iterrows():
    id = str(uuid.uuid4())
    audio_path = row["audio"]
    transcription = row["sentence"]   
    audio_array, sampling_rate = wav_to_array(audio_path)
    ds_item = {
        "path": fix_audio_path(audio_path),
        "id": id,
        "transcription": transcription,
        "audio": {
            "path": fix_audio_path(audio_path),
            "array": audio_array,
            "sampling_rate": sampling_rate
        }
    } 
    ds_val.append(ds_item)

from datasets import Dataset, DatasetDict

big_dict_train = {"path":[], "id":[], "audio":[], "transcription":[]}

for item in ds_train:
    big_dict_train["path"].append(item["path"])
    big_dict_train["id"].append(item["id"])
    big_dict_train["audio"].append(item["audio"])
    big_dict_train["transcription"].append(item["transcription"])

final_ds_train = Dataset.from_dict(big_dict_train)

big_dict_val = {"path":[], "id":[], "audio":[], "transcription":[]}

for item in ds_val:
    big_dict_val["path"].append(item["path"])
    big_dict_val["id"].append(item["id"])
    big_dict_val["audio"].append(item["audio"])
    big_dict_val["transcription"].append(item["transcription"])


final_ds_val = Dataset.from_dict(big_dict_val)


dataset_dict = DatasetDict({
    "train": final_ds_train,
    "validation": final_ds_val
})

# 3. (Optional) Set metadata—description, license, etc.—so your dataset card is populated
dataset_dict.push_to_hub(
    repo_id="WandererGuy/khm_vie_STTT",
)

"""
tutorial 
https://huggingface.co/docs/datasets/en/create_dataset
https://huggingface.co/docs/datasets/en/upload_dataset

"""