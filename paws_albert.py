# !pip install transformers
# !pip install datasets
# !pip install pytorch-lightning

import torch
import pandas as pd
import torch
import torch.nn as nn
import transformers
import torchmetrics
import pytorch_lightning as pl

from datasets import load_dataset,load_metric
dataset = load_dataset("paws", "labeled_final")

CONFIG={
    'sentence1':'sentence1',
    'sentence2':'sentence2',
    'labels':'label',
    'SEED':13,
    'MAX_LEN':256,
    'MODEL_NAME_OR_PATH':'../input/cad-albert-3/cad_albert_v3',
    'LEARNING_RATE':2e-5,
    'ADAM_EPSILON':1e-8,
    'WEIGHT_DECAY':0.0,
    'NUM_CLASSES':2,
    'TRAIN_BS':32,
    'VAL_BS':32,
    'WARMUP_STEPS':0,
    'MAX_EPOCHS':10,
    'CHECKPOINT_DIR':'./checkpoints',
    'NUM_WORKERS':2,
    'PRECISION':16,
    'MODEL_SAVE_NAME':'RUN-1-ALBERT'
}

class PAWSDataset(torch.utils.data.Dataset):

  def __init__(self,max_len:int,tokenizer,dataset):
    super().__init__()
    self.dataset = dataset
    self.max_len = max_len
    self.tokenizer = tokenizer
  
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    sample = self.dataset[idx]
    sentence_1 = sample['sentence1']
    sentence_2 = sample['sentence2']
    encoded_input=self.tokenizer.encode_plus(
        text=sentence_1,
        text_pair=sentence_2,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=self.max_len,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return {
        'labels':torch.tensor(sample['label']),
        'input_ids':encoded_input['input_ids'].view(-1),
        'attention_mask':encoded_input['attention_mask'].view(-1),
        'token_type_ids':encoded_input['token_type_ids'].view(-1),
    }

class PAWSDataModule(pl.LightningDataModule):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset

    def prepare_data(self):
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME_OR_PATH'])

    def setup(self, stage):

      if stage=='fit':
        self.train_ds,self.valid_ds = self.dataset['train'],self.dataset['validation']

    def get_dataset(self,df):
      dataset = PAWSDataset(max_len=CONFIG['MAX_LEN'],
                               tokenizer=self.tokenizer, dataset = df)
      return dataset

    def train_dataloader(self):
      train_dataset = self.get_dataset(self.train_ds)
      train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                     batch_size=CONFIG['TRAIN_BS'], 
                                                     shuffle=True, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return train_dataloader

    def val_dataloader(self):
      val_dataset = self.get_dataset(self.valid_ds)
      val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                     batch_size=CONFIG['VAL_BS'], 
                                                     shuffle=False, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return val_dataloader

    def test_dataloader(self):
      test_dataset=self.get_dataset(self.test_df)
      test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                     batch_size=CONFIG['VAL_BS'], 
                                                     shuffle=False, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return test_dataloader

class PAWSFineTuningModel(pl.LightningModule):

  def __init__(self,model_name_or_path:str,
               num_labels:int,
               learning_rate:float,
               adam_epsilon:float,
               weight_decay:float,
               max_len:int,
               warmup_steps:int,
               gpus:int,max_epochs:int,
               accumulate_grad_batches:int,
               total_steps:int):
    super().__init__()
    self.model_name_or_path=model_name_or_path
    self.num_labels=num_labels
    self.save_hyperparameters('learning_rate','adam_epsilon','weight_decay','max_len','gpus','accumulate_grad_batches','max_epochs','warmup_steps') 
    self.config = transformers.AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
    self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy()
      ]
    )
    self.train_metrics=metrics.clone()
    self.val_metrics=metrics.clone()
    self.total_steps = total_steps


  def forward(self,inputs):
    return self.model(**inputs)
  
  def training_step(self,batch,batch_idx):
    loss,logits=self(batch)[:2]
    predictions=torch.argmax(logits,dim=1)
    self.train_metrics(predictions,batch['labels'])
    self.log_dict({'train_accuracy':self.train_metrics['Accuracy'],}, on_step=False, on_epoch=True)
    return {
        'loss':loss,
        'predictions':predictions,
        'labels':batch['labels']
    }
  
  def validation_step(self,batch,batch_idx):
    loss,logits=self(batch)[:2]
    predictions=torch.argmax(logits,dim=1)
    self.val_metrics(predictions,batch['labels'])
    self.log_dict({'val_accuracy':self.val_metrics['Accuracy'],}, on_step=False, on_epoch=True)
    return {
        'loss':loss,
        'predictions':predictions,
        'labels':batch['labels']
    }
  

  def validation_epoch_end(self,outputs):
    loss=torch.tensor([x['loss'] for x in outputs])
    loss = loss.mean()
    self.log('val_loss', loss, prog_bar=True,on_step=False, on_epoch=True )
  
  def training_epoch_end(self,outputs):
    loss=torch.tensor([x['loss'] for x in outputs])
    loss = loss.mean()
    self.log('train_loss', loss, prog_bar=True,on_step=False, on_epoch=True )
  
  def configure_optimizers(self):
    model = self.model
    no_decay = ["bias", "LayerNorm.weight","LayerNorm.bias"]
    optimizer_grouped_parameters = [
          {
              "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": self.hparams.weight_decay,
          },
          {
              "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
    )
    scheduler = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
    }
    return [optimizer] ,[scheduler]

model_save_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath=CONFIG['CHECKPOINT_DIR'],
    filename=f"{CONFIG['MODEL_SAVE_NAME']}"+'-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                     max_epochs=CONFIG['MAX_EPOCHS'],
                     callbacks=[model_save_checkpoint,pl.callbacks.EarlyStopping(monitor="val_loss")],
                     precision=CONFIG['PRECISION'],
                     num_sanity_val_steps=0
                    )

total_steps = len(dataset['train'])//(CONFIG['TRAIN_BS'])//(trainer.accumulate_grad_batches*float(CONFIG['MAX_EPOCHS']))

model=PAWSFineTuningModel(
    model_name_or_path=CONFIG['MODEL_NAME_OR_PATH'],
    num_labels=CONFIG['NUM_CLASSES'],
    learning_rate=CONFIG['LEARNING_RATE'],
    adam_epsilon=CONFIG['ADAM_EPSILON'],
    weight_decay=CONFIG['WEIGHT_DECAY'],
    max_len=CONFIG['MAX_LEN'],
    warmup_steps=CONFIG['WARMUP_STEPS'],
    max_epochs=trainer.max_epochs,
    gpus=trainer.gpus,
    accumulate_grad_batches=trainer.accumulate_grad_batches,
    total_steps = total_steps
)

paws_dm = PAWSDataModule(dataset)
trainer.fit(model,paws_dm)

trainer.validate()