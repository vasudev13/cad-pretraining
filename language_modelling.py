# !git clone https://github.com/huggingface/transformers
# !pip install .
# !pip install datasets

import os
from google.colab import drive
import pandas as pd

drive.mount('/gdrive')

os.chdir('/content/')

original_text = pd.read_csv('/gdrive/MyDrive/imbdb_original_v3.csv')
perturbed_text = pd.read_csv('/gdrive/MyDrive/imbdb_perturbed_v9.csv')

original_text.shape

perturbed_text.shape

perturbed_text.head()

text = []
columns = perturbed_text.columns
for index, row in perturbed_text.iterrows():
  for column in columns:
    if not pd.isnull(row[column]):
      text.append(row[column])

perturbed_df = pd.DataFrame()
perturbed_df['Text'] = text

data = perturbed_df.append(original_text)
data = data.sample(frac=1).reset_index(drop=True) # Shuffle
last_index = int(0.9*len(data))
train_data = data[:last_index]
val_data = data[last_index:]

textfile = open("imdb_train.txt", "w")
for text in train_data['Text'].values.tolist():
  textfile.write(text + "\n")
textfile.close()

textfile = open("imdb_val.txt", "w")
for text in val_data['Text'].values.tolist():
  textfile.write(text + "\n")
textfile.close()

!python transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path albert-base-v2 \
    --train_file /content/imdb_train.txt \
    --validation_file /content/imdb_val.txt \
    --do_train \
    --do_eval \
    --output_dir cad_albert_v3 \
    --fp16 \
    --dataloader_num_workers 8\
    --save_steps 1000 \
    --line_by_line

!cp -r /content/cad_albert_v3/ /gdrive/MyDrive/