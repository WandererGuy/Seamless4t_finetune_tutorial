finetune seamless4t is extremely hard but rewarding , here I finetune on my own custom dataset , enjoy <br>
U can skip to step 2 if you want to finetune on my demo dataset only 
this is how I am able to finetune seamless4t , you can take inspiration from this and do with your own case; here I finetune seamless4t_medium model on RTX 3090(24GB VRAM) for SPEECH_TO_TEXT TASK (model during finetuning takes up nearly 19GB)
- step 1: I create a dataset like FLEURS then push to huggingface dataset hub , I have a csv contains local dataset (using my script create_fleurs_dataset.py here)
- step 2: I load those dataset and save dataset value in ./huggingface_cache for later load
(using script load_save_train.py and load_save_val.py)
- step 3: I modify seamless4t repository script to enable it to load my dataset value (which I stored in last step) instead of google/fleurs dataset  <br>
how did i do it , I have to modify this script in src/seamless_communication/datasets/huggingface.py, example script I fixed I put in this huggingface.py here in the old script place
- step 4: those script updates so you need to abandon existing environment and create new envinment then ```pip install .``` like in seamless4t installation guide (each time you update seamless4t script , you must create environment or else the modification to code wont take effect)
- step 5: I follow like seamless4t finetune guide, my example commands are
```
export DATASET_DIR=./khmer_vietnamese_dataset_v2
mkdir -p $DATASET_DIR

python src/seamless_communication/cli/m4t/finetune/dataset.py \
  --name google/fleurs \
  --source_lang vie \
  --target_lang vie \
  --split train \
  --save_dir $DATASET_DIR

python src/seamless_communication/cli/m4t/finetune/dataset.py \
  --name google/fleurs \
  --source_lang vie \
  --target_lang vie \
  --split validation \
  --save_dir $DATASET_DIR


  python src/seamless_communication/cli/m4t/finetune/finetune.py \
   --mode SPEECH_TO_TEXT \
   --train_dataset $DATASET_DIR/train_manifest.json  \
   --eval_dataset $DATASET_DIR/validation_manifest.json \
   --learning_rate 1e-6 \
   --warmup_steps 100 \
   --max_epochs 50 \
   --patience 50 \
   --model_name seamlessM4T_medium \
   --save_model_to $DATASET_DIR/checkpoint.pt \
   --batch_size 1
```
step 6: use seamless4t finetune checkpoint , i use script inference.py here
