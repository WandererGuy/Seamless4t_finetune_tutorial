from datasets import load_dataset
cache_dir = "hugging_face_cache"
ds_train = load_dataset("WandererGuy/khm_vie_STTT", split="train", cache_dir=cache_dir)
from datasets import Dataset, DatasetDict

big_dict_train = {"path":[], "id":[], "audio":[], "transcription":[]}

import soundfile as sf
import os 
save_dataset = "save_dataset"
from tqdm import tqdm
for item in tqdm(ds_train, total=len(ds_train)):
    
    audio_array = item["audio"]["array"]
    sampling_rate = item["audio"]["sampling_rate"]
    out_path = item["path"]
    out_path = os.path.join(save_dataset, out_path)
    big_dict_train["path"].append(out_path)
    big_dict_train["id"].append(item["id"])
    big_dict_train["audio"].append(item["audio"])
    big_dict_train["transcription"].append(item["transcription"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        continue
    sf.write(out_path, audio_array, sampling_rate)

final_ds_train = Dataset.from_dict(big_dict_train)
final_ds_train.save_to_disk(os.path.join(cache_dir, "train"))
