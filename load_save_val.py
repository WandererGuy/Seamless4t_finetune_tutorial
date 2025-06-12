from datasets import load_dataset
cache_dir = "hugging_face_cache"
ds_val = load_dataset("WandererGuy/khm_vie_STTT", split="validation", cache_dir=cache_dir)
from datasets import Dataset, DatasetDict

big_dict_val = {"path":[], "id":[], "audio":[], "transcription":[]}

import soundfile as sf
import os 
save_dataset = "save_dataset"
from tqdm import tqdm
for item in tqdm(ds_val, total=len(ds_val)):
    audio_array = item["audio"]["array"]
    sampling_rate = item["audio"]["sampling_rate"]
    out_path = item["path"]
    out_path = os.path.join(save_dataset, out_path)
    big_dict_val["path"].append(out_path)
    big_dict_val["id"].append(item["id"])
    big_dict_val["audio"].append(item["audio"])
    big_dict_val["transcription"].append(item["transcription"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        continue
    sf.write(out_path, audio_array, sampling_rate)

final_ds_val = Dataset.from_dict(big_dict_val)
final_ds_val.save_to_disk(os.path.join(cache_dir, "val"))
