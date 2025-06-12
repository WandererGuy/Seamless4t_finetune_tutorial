import torch
from seamless_communication.inference import Translator

saved_finetune_ckpt_path = "khmer_vietnamese_dataset_v2/checkpoint.pt"
finetuned_checkpoint = torch.load(saved_finetune_ckpt_path)["model"]
model_name = torch.load(saved_finetune_ckpt_path)["model_name"]
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda:0"),
    dtype=torch.float16 
)

# this 2 line only applies for SPEECH TO TEXT , see main function ịn src/seamless_communication/cli/m4t/finetune/finetune.py to see 
translator.model.text_encoder = None
translator.model.t2u_model = None

model_dict = translator.model.state_dict()
translator.model.load_state_dict(finetuned_checkpoint, strict=False)

path_to_input_audio = "finetune_dataset/wav_files/967f4bd2-7864-4e98-9eb3-9c7e20f8385b.wav"
tgt_lang = "vie"
# S2TT
text_output, _ = translator.predict(
    input=path_to_input_audio,
    task_str="S2TT",
    tgt_lang=tgt_lang
)
print (text_output[0])
# ASR
# This is equivalent to S2TT with `<tgt_lang>=<src_lang>`.
text_output, _ = translator.predict(
    input=path_to_input_audio,
    task_str="ASR",
    tgt_lang=tgt_lang
)
print (text_output[0])
